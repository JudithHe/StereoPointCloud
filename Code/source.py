from testFunctionality import *


# The addresses of the input images
INPUT_LEFT = "im0.png"
INPUT_RIGHT = "im1.png"

# The distance between the two cameras, taken from calib.txt
CAMERA_DISTANCE = 1438.004 - 1263.818

# The focal length of the two cameras, taken from calib.txt
FOCAL_LENGTH = 5299.313

# The maximum distance used for extrapolating the depth of non-feature points, in terms of fraction of picture width
EXTRAPOLATION_FRACTION = 0.05

# The size to which the raw picture will be changed, to increase the speed of calculations
RESIZE_WIDTH = 600
RESIZE_HEIGHT = 600


def run_sift(image_left_resized, image_right_resized):
    print("Running the sift function...")

    print("\tSift initializing...")
    # https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    print("\tKeypoints and Descriptors detection...")
    # find the keypoints and descriptors with SIFT
    keypoint_left, descriptor_left = sift.detectAndCompute(image_left_resized, None)
    keypoint_right, descriptor_right = sift.detectAndCompute(image_right_resized, None)

    print("\tBrute-Force Matcher...")
    # Brute-Force Matcher with default parameters
    bruteforce = cv2.BFMatcher()
    matches = bruteforce.knnMatch(descriptor_left, descriptor_right, k=2)

    print("\tRatio Test...")
    # Applying ratio test
    good = []
    for m, n in matches:
        if m.distance <= 0.8 * n.distance:
            good.append(m)
    return [keypoint_left, keypoint_right, good]


def draw_matches(image_left_resized, image_right_resized, keypoint_left, keypoint_right, good, x=None, y=None, w=None,
                 h=None):
    print('Viewing results...')

    if x is None:
        x = 0
        y = 0
        w = 1200
        h = 600

    # cv.drawMatchesKnn expects list of lists as matches.
    # img_results = cv2.drawMatchesKnn(image_left_raw, keypoint_left, image_right_raw, keypoint_right, good, None,
    #                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image_left_gray = cv2.cvtColor(image_left_resized, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right_resized, cv2.COLOR_BGR2GRAY)

    print('\tPress any key to quit...')
    drawMatches(image_left_gray, keypoint_left, image_right_gray, keypoint_right, good, x, y, w, h)

    return True


def calculate_depth(image_left_resized, image_right_resized, keypoint_left, keypoint_right, good):
    print("Calculating the depth of matched keypoints...")

    # Will store the depth of each matched keypoint in depth
    depth = np.zeros(len(good))

    # The coordinates of the centre point of each picture, 1=left, 2=right
    centre1 = np.array(np.array(np.shape(image_left_resized)[0:1]) / 2, dtype=int)
    centre2 = np.array(np.array(np.shape(image_right_resized)[0:1]) / 2, dtype=int)

    maxed = []
    for i in range(len(good)):
        match = good[i]

        idx1 = match.queryIdx
        idx2 = match.trainIdx

        (x1, y1) = np.array(keypoint_left[idx1].pt, dtype=int) - centre1
        (x2, y2) = np.array(keypoint_right[idx2].pt, dtype=int) - centre2

        if abs(y1 - y2) > min(centre1[0], centre2) / 100 or x1 <= x2:
            depth[i] = -1
        else:
            depth[i] = CAMERA_DISTANCE * FOCAL_LENGTH / abs(x1 - x2)

    max_val = depth.max()
    for i in maxed:
        depth[i] = max_val

    average = depth[depth > -1].sum() / len(depth[depth > -1])
    variance = depth[depth > -1].var()

    depth_filtered = []
    reference_left = []
    reference_right = []
    for i in range(len(good)):
        if (depth[i] == -1) or (depth[i] >= average + 3 * math.sqrt(variance)) or (depth[i] <= average - 3 *
                                                                                   math.sqrt(variance)):
            continue

        depth_filtered += [depth[i]]
        match = good[i]

        idx1 = match.queryIdx
        idx2 = match.trainIdx

        reference_left += [[keypoint_left[idx1].pt[0], keypoint_left[idx1].pt[1]]]
        reference_right += [[keypoint_right[idx2].pt[0], keypoint_right[idx2].pt[1]]]

    mindepth = np.array(depth_filtered).min()
    maxdepth = np.array(depth_filtered).max()
    return [np.array(reference_left, dtype=int), np.array(reference_right, dtype=int), (depth_filtered - mindepth) /
            (maxdepth - mindepth)]


def depth_to_color(z):
    color = [z * 255, 0, (1 - z) * 255]

    return color


def show_feature_depth(image_left_resized, image_right_resized, reference_left, reference_right, depth):
    print("Viewing the depth of each feature...")
    for i in range(len(reference_left)):
        [x, y] = reference_left[i]
        draw_circle(image_left_resized, x, y, 5, depth_to_color(depth[i]))

        [x, y] = reference_right[i]
        draw_circle(image_right_resized, x, y, 5, depth_to_color(depth[i]))

    show_image('Left', image_left_resized, 0, 0, 600, 600)
    show_image('Right', image_right_resized, 600, 0, 600, 600)

    cv2.waitKey(0)

    cv2.destroyWindow('Left')
    cv2.destroyWindow('Right')

    return True


def extrapolate_depth(image_left_resized, image_right_resized, reference_left, reference_right, depth):
    print("Extrapolating the depth of all points...")

    image_left_depth = np.copy(image_left_resized)
    image_right_depth = np.copy(image_right_resized)

    w = image_left_depth.shape[1]
    h = image_left_depth.shape[0]

    for i in range(0, h, 5):
        print('\t%d/%d' % (i, h))
        for j in range(0, w, 5):
            numerator = -1
            denominator = 0

            for k in range(len(reference_left)):
                (x, y) = reference_left[k]
                distance = math.sqrt((x - j)**2 + (y-i)**2)

                if distance > max(w, h) * EXTRAPOLATION_FRACTION:
                    continue
                elif numerator == -1:
                    numerator = 0

                if x == j and y == i:
                    numerator = depth[k]
                    denominator = 0
                    break
                numerator += depth[k] / distance
                denominator += 1 / distance

            if denominator == 0:
                if numerator == -1:
                    d = 1
                else:
                    d = numerator
            else:
                d = numerator / denominator

            draw_circle(image_left_depth, j, i, 2, depth_to_color(d))

    return [image_left_depth, image_right_depth]


def show_extrapolated_depth(image_left_depth, image_right_depth):
    print("Showing interpolated depths...")

    show_image("Left", image_left_depth, 0, 0, 600, 600)
    # show_image("Right", image_right_depth, 600, 0, 600, 600)
    cv2.waitKey(0)

    return True


def main():
    print("Loading images...")
    # Reading the two pictures
    image_left_raw = cv2.imread(INPUT_LEFT)
    image_right_raw = cv2.imread(INPUT_RIGHT)

    print("Resizing the images...")
    # Resizing the two pictures to increase speed of analysis
    image_left_resized = cv2.resize(image_left_raw, (RESIZE_WIDTH, RESIZE_HEIGHT))
    image_right_resized = cv2.resize(image_right_raw, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Blurring the two pictures
    # image_left_resized = cv2.blur(image_left_resized, (5, 5))
    # image_right_resized = cv2.blur(image_right_resized, (5, 5))

    # Using sift to detect keypoints in both images and matching them using a bruteforce matcher
    [keypoint_left, keypoint_right, good] = run_sift(image_left_resized, image_right_resized)

    # Show the matched keypoints
    draw_matches(image_left_resized, image_right_resized, keypoint_left, keypoint_right, good)

    # Will calculate the depth of each matched keypoint
    [reference_left, reference_right, depth] = calculate_depth(image_left_resized, image_right_resized, keypoint_left,
                                                 keypoint_right, good)

    # Drawing the depth of each matched keypoint
    show_feature_depth(image_left_resized, image_right_resized, reference_left, reference_right, depth)

    # Extapolating the depth of all points, and showing it
    [image_left_depth, image_right_depth] = extrapolate_depth(image_left_resized, image_right_resized, reference_left,
                                               reference_right, depth)
    show_extrapolated_depth(image_left_depth, image_right_depth)

    print("Exiting...")
    return True


if __name__ == "__main__":
    main()
