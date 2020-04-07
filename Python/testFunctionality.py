import matplotlib.pyplot as plt
import math
import random
import numpy as np
import cv2


# Connects the matching keypoints in two images
def drawMatches(img1, kp1, img2, kp2, matches, x=None, y=None, w=None, h=None):
    if x is None:
        x = 0
        y = 0
        w = 600
        h = 600

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # Create the output image
    # The rows of the output are the largest between the two images
    # and the columns are simply the sum of the two together
    # The intent is to make this a colour image, so make this 3 channels
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1

        blue = random.randint(0, 255)
        green = random.randint(0, 255)
        red = random.randint(0, 255)

        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (blue, green, red), 1)


    # Show the image
    show_image('Matched Features', out, x, y, w, h)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


# Presents an image at (x,y) in a window with size (w,h)
def show_image(title, image, x=None, y=None, w=None, h=None):
    if x is None:
        x = 0
        y = 0
        w = 960
        h = 540

    cv2.namedWindow(title)
    cv2.moveWindow(title, x, y)

    image_resized = cv2.resize(image, (w, h))
    cv2.imshow(title, image_resized)

    return True


# Draws a circle on an image
def draw_circle(image, x, y, r, color=None):
    if color is None:
        color = [1, 0, 0]

    w = np.shape(image)[1]
    h = np.shape(image)[0]

    for i in range(max(0, y - r), min(h, y + r)):
        for j in range(max(0, x - r), min(w, x + r)):
            if (i - y) ** 2 + (j - x) ** 2 <= r ** 2:
                image[i][j] = color

    return True
