<h1>Overview</h1>

This code uses two pictures taken from two cameras aligned on the horizon to generate a 3D stereo point cloud of an environment.
There are two components to this code: the Python depth calculation code and the c++ point cloud construction code.

A detailed explanation of the algorithm is described in <b>report.pdf</b> and an overview of the code is given in the video <b>Video Introduction.mp4</b>.

<h2>Python: Depth Calculation and 2D Depth Map Construction</h2>
The python code is available in the <i>Python</i> directory. There are three 4 python files:
<ul>
  <li>source.py - Generates a 2D depth map of the environment, while running the SIFT algorithm on the entirety of the picture at once.</li>
  <li>source_detail.py - Generates a 2D depth map of the environment, but unlike source.py the SIFT algorithm is run on small segments of the picture and the output of the algorithm is accumulated to make the 2D depth map.</li>
  <li>source_generator.py - The same as source.py, but instead of generating a 2D depth map it generates an output text file <b>points.txt</b> which can be processed by the C++ code to generate the 3D point cloud.</li>
  <li>testFunctionality.py - Contains a set of functions used in the other three files</li>
</ul>

<h2>C++: 3D Point Cloud Construction</h2>
Place the output of <b>source_generator.py</b>, namely <i>points.txt</i> next to the file <b>source.cpp</b>. Then, execute the code and a 3D point cloud is generated.

Use the arrow keys to rotate the camera and view the point cloud from different perspectives.
