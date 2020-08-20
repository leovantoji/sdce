# Udacity Self-Driving Car Engineer Nanodegree
## Computer Vision Fundamentals
- From [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html) website, **Canny Edge Detection** is a popular edge detection algorithm. It was developed by John F. Canny in 1986. It is a **multi-stage algorithm**.
  - **Noise reduction**: Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter.
  - **Finding Intensity Gradient of the image**: Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction to get first derivative in horizontal direction *G<sub>x</sub>* and vertical direction *G<sub>y</sub>*. From these two images, we can find edge gradient and direction for each pixel as follows. Gradient direction is always perpendicular to edges. It is rounded to one of four angles representing vertical, horizontal and two diagonal directions.  
    ![canny_edge_intensity_gradient](https://github.com/leovantoji/sdce/blob/master/images/canny_edge_intensity_gradient.png)
  - **Non-maximum Suppression**: After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted pixels which may not constitute the edge. For this, at every pixel, pixel is checked if it is a local maximum in its neighbourhood in the direction of the gradient. Point A is on the edge (in vertical direction). Gradient direction is normal to the edge. Point B and C are in gradient directions. So point A is checked with point B and C to see if it forms a local maximum. If so, it is considered for next stage, otherwise, it is suppressed (put to zero). In short, the result you get is a binary image with "thin edges".  
  ![canny_edge_non_maximum_suppression](https://github.com/leovantoji/sdce/blob/master/images/canny_edge_non_maximum_suppression.jpg)
  - **Hysteresis Thresholding**: This stage decides which edges are really edges and which edges are not. For this, we need two threshold values, *minVal* and *maxVal*. Any edges with intensity gradient more than *maxVal* are sure to be edges and those below *minVal* are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. If they are connected to *sure-edge* pixels, they are considered to be part of edges. Otherwise, they are also discarded. The edge A is above the *maxVal* so it is considered a *sure-edge*. Although edge C is below *maxVal*, it is also considered as a valid edge since it is connected to edge A. Edge B is not an edge since it is not connected to any *sure-edge* despite having its value higher than *minVal*. This stage also removes small pixels noises on the assumption that edges are long lines.  
  ![canny_edge_hysteresis_thresholding](https://github.com/leovantoji/sdce/blob/master/images/canny_edge_hysteresis_thresholding.jpg)
- Canny Edge Detection example:
  - Read an image:
  ```python
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  image = mpimg.imread('exit-ram.jpg')
  plt.imshow(image)
  ```
  - Convert image to grayscale
  ```python
  import cv2
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # grayscale conversion
  plt.imshow(gray, cmap='gray')
  ```
  - Canny Edge detection from OpenCV. In this case, you're applying `Canny` function to the image `gray`, and your output will be another image called `edges`. As far as a ratio of `low_threshold` to `high_threshold`, John Canny himself recommended a low to high ratio of **1:2** or **1:3**. What would make sense as a reasonable range for these parameters? In our case, converting to grayscale has left us with an 8-bit image, so each pixel can take *2<sup>8</sup> = 256* possible values. Ergo, the pixel values range from 0 to 255. This range implies that derivatives (essentially, the value differences from pixel to pixel) will be **on the scale of tens or hundreds**. 
  ```python
  edges = cv2.Canny(gray, low_threshold, high_threshold)
  ```
- **Hough Transform**: In image space, a line is plotted as x vs. y, but in 1962, Paul Hough devised a method for representing lines in parameter space, which we will call "Hough space" in his honour. In Hough space, I can represent my "x vs. y" **line as a point** in "m vs. b" instead. The Hough Transform is just the conversion from image space to Hough space.
  - Parallel lines in Image space to Hough space: Option C  
  ![22-q-hough-intro-quiz](https://github.com/leovantoji/sdce/blob/master/images/22-q-hough-intro-quiz.png)
  - A point in Image space to Hough space: Option A  
  ![23-q-hough-second-quiz](https://github.com/leovantoji/sdce/blob/master/images/23-q-hough-second-quiz.png)
  - 2 points in Image space to Hough space: Option C  
  ![25-q-hough-fourth-quiz-updated2](https://github.com/leovantoji/sdce/blob/master/images/25-q-hough-fourth-quiz-updated2.png)
  - A square in Image space to Hough space: Option C  
  ![26-hough-quiz](https://github.com/leovantoji/sdce/blob/master/images/26-hough-quiz.png)
  - Nonetheless, the slope *m* is undefined when the line is vertical. Therefore, we need to use another parameter space: *ρ = xcosθ + ysinθ*. *ρ* (rho) is the distance of the line from the origin, and *θ* are the angle away from the horizontal.   
  ![hough_space_polar](https://github.com/leovantoji/sdce/blob/master/images/hough_space_polar.png)
- Implementing a Hough Transform on Edge Detected Image:
  - [From Scratch](https://alyssaq.github.io/2014/understanding-hough-transform/).
  - Using OpenCV function `HoughLinesP`.
    ```python
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    ```
    - `mask_edges` are the outputs from `Canny` function.
    - `lines` is an array containing end points *(x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>, y<sub>2</sub>)* of all line segments detected by the transform operation. 
    - `rho` and `theta` are specified in units of pixels and radians respectively. 
    - The `threshold` parameter specifies the minimum number of votes (intersections in a given grid cell) a candidate line needs to have to make it to the output. 
    - The empty `np.array([])` is just a placeholder. `
    - `min_line_length` is the minimum length of a line (in pixels) that you will accept in the output.
    - `max_line_gap` is the maximum distance (again, in pixels) between segments that you will allow to be connected into a single line.

## Camera Calibration
- **Image distortion** occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image. This transformation isn't perfect because:
  - Distortion can change the apparent size and shape of an object in an image.
  - Distortion can cause an object's appearance to change depending on where it is in the field of view.
  - Distortion can make objects appear closer or farther away than they actually are.
- Thus, the first step in analysing camera image, is to **undo this distortion** so that you can get correct and useful information about them.  
  ![distorted_images](https://github.com/leovantoji/sdce/blob/master/images/distorted_images.png)
- Real cameras use **curved lenses** to form an image, and light rays often bend a little too much or too little at the edges of these lenses. This creates an effect that **distorts the edges of images**, so that **lines or objects appear more or less curved than they actually are**. This is called **radial distortion**, and it's the most common type of distortion.
- Another distortion type called **tangential distortion**, occurs when a camera's lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is. This makes an image **look tilted** so that some **objects appear farther away or closer than they actually are**.
- There are 3 coefficients needed to correct for **radial distortion: k<sub>1</sub>, k<sub>2</sub>** and **k<sub>3</sub>**. *Note*: The distortion coefficient **k<sub>3</sub>** is required to accurately reflect *major* radial distortion (like in wide angle lenses). However, for minor radial distortion, which most regular camera lenses have, **k<sub>3</sub>** has a value close to or equal to 0 and is negligible. Thus, in OpenCV, you may choose to ignore this coefficient, and this is also the reason why **k<sub>3</sub>** apppears at the end of the distortion values array: `[k1, k2, p1, p2, k3]`.  
  ![correct_radial_distortion](https://github.com/leovantoji/sdce/blob/master/images/correct_radial_distortion.png)
- Formula for radial distortion correction: 
  - *x<sub>distorted</sub> = x<sub>ideal</sub>(1 + k<sub>1</sub>r<sup>2</sup> + k<sub>2</sub>r<sup>4</sup> + k<sub>3</sub>r<sup>6</sup>)*
  - *y<sub>distorted</sub> = y<sub>ideal</sub>(1 + k<sub>1</sub>r<sup>2</sup> + k<sub>2</sub>r<sup>4</sup> + k<sub>3</sub>r<sup>6</sup>)*
- Formula for tangential distortion correction:
  - *x<sub>corrected</sub> = x + \[2p<sub>1</sub>xy + p<sub>2</sub>(r<sup>2</sup> + 2x<sup>2</sup>)\]*
  - *y<sub>corrected</sub> = y + \[p<sub>1</sub>(r<sup>2</sup> + 2y<sup>2</sup>) + 2p<sub>2</sub>xy\]*
- **Pictures of known shapes** can help us **correct any distortion errors**. A **chessboard** is normally used to calibrate the camera because its **regular high contrast pattern** makes it easy to detect automatically. Therefore, multiple pictures of a chessboard against a flat surface can be used. OpenCV has 2 useful functions for this purpose.
  - `findChessboardCorners()`: can automatically find corners in an image of a chessboard pattern.
  - `drawChessboardCorners()`: can automatically draw corners in an image of a chessboard pattern.  
  ![corners-found3](https://github.com/leovantoji/sdce/blob/master/images/corners-found3.jpg)  
- Two main steps for camera calibration:
  - Use chessboard images to obtain image points (2D) and object points (3D).
  - Use OpenCV functions `cv2.calibrateCamera()` and `cv2.undistort()` to compute the calibration and undistortion.
  ```python
  # codes from https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
  
  # prepare object points, like (0,0,0), (1,0,0), ..., (7,5,0)
  nx, ny = 8, 6 # number of inside corners for each row and column (based on the image)
  objp = np.zeros((ny*nx,3), np.float32)
  objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
  
  # arrays to store object and image points from all images
  objpoints = [] # 3D points in real world space
  imgpoints = [] # 2D points in image plane
  
  # make a list of calibration images
  images = glob.glob('calibration_wide/GO*.jpg')
  
  # step through the list and search for chessboard corners
  for _, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # add object/image points if found
    if (ret):
      objpoints.append(objp)
      imgpoints.append(corners)
      
      # draw and display the corners
      cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
      cv2.imshow('img', img)
      cv2.waitKey(500)
      
  cv2.destroyAllWindows()
  
  # test undistortion on an image
  img = cv2.imread('calibration_wide/test_image.jpg')
  
  # perform camera calibration given object points and image points
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
  
  dst = cv2.undistort(img, mtx, dist, None, mtx)
  cv2.imwrite('calibration_wide/test_undist.jpg', dst)
  ```
|Original Image|Undistorted Image|
|:-:|:-:|
|![orig-and-undist](https://github.com/leovantoji/sdce/blob/master/images/orig-and-undist.png)|![orig-and-undist2](https://github.com/leovantoji/sdce/blob/master/images/orig-and-undist2.png)|
- 




