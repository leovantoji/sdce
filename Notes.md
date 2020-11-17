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
  ![orig-and-undist](https://github.com/leovantoji/sdce/blob/master/images/orig-and-undist.png)
  
- Self-driving cars need to be told the correct steering angle to turn left or right. This angle can be calculated based on the speed and dynamics of the car and the curvature of the lane. One way to **calculate the curvature of a lane line**, is to fit a 2<sup>nd</sup> degree polynomial to that line, and from this, you can easily extract useful information. For a lane line that is close to vertical, you can fit a line using this formula: *f(y) = Ay<sup>2</sup> + By + C*.
  - *A*: the curvature of the lane line.
  - *B*: the heading or direction that the line is pointing.
  - *C*: the position of the line based on how far away it is from the very left of an image (*y = 0*).
- In an image, **perspective** is the phenomenon where an object appears smaller the farther away it is from a particular viewpoint, and parallel lines appear to converge to a point. A **perspective transform** maps the points in a given image to **different, desired, image points** with a new perspective. A **bird's-eye view transform**, which allows us to view a lane from above, is extremely **useful for calculating the lane curvature**.  
  ![birds_eye_view_transform](https://github.com/leovantoji/sdce/blob/master/images/birds_eye_view_transform.png)
- OpenCV's useful functions:
  - Compute the perspective transform, M, given source and destination points (need 4 points):
  ```python
  M = cv2.getPerspectiveTransform(src, dst)
  ```
  - Compute the inverse perspective transform:
  ```python
  Minv = cv2.getPerspectiveTransform(dst, src)
  ```
  - Warp an image using the perspective transform, M:
  ```python
  warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
  ```

## Gradient and Colour Space
- **Sobel operator** is at the **heart of the Canny edge dection algorithm**. Applying the Sobel operator to an image is a **way of taking the derivative of the image** in the *x* or *y* direction. Below are examples of the operators witth a **kernel size** of 3 for *Sobel<sub>x</sub>* and *Sobel<sub>y</sub>*. 3x3 is the minimum size, and the kernel size can be any **odd number**. A **larger kernel** implies taking the gradient over a larger region of the image, or, in other words, a smoother gradient.  
  ![sobel-operator](https://github.com/leovantoji/sdce/blob/master/images/sobel-operator.png)
- If the image is flat across that region (i.e., there is little change in values across the given region), then the result (the sum of the element-wise product of the operator and corresponding immage pixels) will be zero.  
  ![Sobel_flat_region_example](https://github.com/leovantoji/sdce/blob/master/images/Sobel_flat_region_example.png)
- If *S<sub>x</sub>* operator is applied to a region of the image where values are rising from left to right, then the result will be positive, implying a positive derivative. The sum of this matrix is 8, meaning a gradient exists in the x-direction.  
  ![Sobel_positive_gradient_example](https://github.com/leovantoji/sdce/blob/master/images/Sobel_positive_gradient_example.png)
- Taking the gradient in the **x-direction emphasises edges closer to vertical**, while taking the gradient in the **y-direction emphasises edges closer to horizontal**.
- Useful functions:
  - NOTE: Convert the image to grayscale as we need to pass a single colour channel to the `cv2.Sobel()` function.
  - Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction):
  ```python
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
  ```
  - Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):
  ```python
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
  ```
  - Calculte the absolute value of the x derivative:
  ```python
  abs_sobelx = np.absolute(sobelx)
  ```
  - Convert the absolute value image to 8-bit:
  ```python
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
  ```
  - Create a mask 1's where the scaled gradient magnitude is `> thresh_min` and `< thresh_max`
  ```python
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1
  ```
- The magnitude, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients. For **a gradient in both the x and y direction**, the magnitude is the square root of the sum of the squares.  
  ![sobel_magnitude_formula](https://github.com/leovantoji/sdce/blob/master/images/sobel_magnitude_formula.png)
- The **direction of the gradient** is simply the **inverse tangent (arctangent) of the y gradient divided by the x gradient**: *arctan(Sobel<sub>y</sub>/Sobel<sub>x</sub>)*. Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of *−π/2* to *π/2*. An orientation of 0 implies a vertical line and orientations of *+/−π/2* imply horizontal lines. 
  - `np.arctan2` can return values between *+/−π*. Nonetheless, as we'll take the absolute value of *Sobel<sub>x</sub>*, this restricts the values to *+/−π/2*.
- **RGB Thresholding** works best on **white lane pixels**, and **doesn't work well** in images with **varying light conditions** or when **lanes are a different colour like yellow**.
- A **colour space** is a **specific organisation of colours**; colour spaces provide a way to categorise colours and represent them in digital images. **RGB** is red-green-blue colour space.   
  ![rgb_color_space](https://github.com/leovantoji/sdce/blob/master/images/rgb_color_space.png)
- Other commonly used colour spaces in image analysis include **HSV (hue, saturation, value)** and **HLS (hue, lightness, saturation)**.
  - **Hue**: the value that **represents the colour** independent of any change in brightness. For instance, light red and dark red have the same hue.
  - **Saturation**: is a **measurement of colourfulness**. Thus, as colours get lighter and closer to white, they have lower a saturation value. On the other hand, the most intense colours like a bright primary colour, have a high saturation value.
  - **Lightness** and **Value**: represent different ways to **measure the relative lightness or darkness of a colour**. For example, a dark red will have a similar hue but much lower value for lightness than a light red.  
  ![hsv_hls](https://github.com/leovantoji/sdce/blob/master/images/hsv_hls.png)
- Useful OpenCV's function:
  ```python
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  ```
- The **S** channel is **preferrable** when it comes to **changing conditions** in image.
  ```python
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  import numpy as np
  import cv2
  
  image = mpimg.imread('test6.jpg')
  
  def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    # 3) Return a binary image of threshold result
    return binary_output
  ```
- Combining **gradient threshold** and **S channel threshold**.  
  ![combine_gradient_s_thresh](https://github.com/leovantoji/sdce/blob/master/images/combine_gradient_s_thresh.png)
  ```python
  def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary
  ```

## Advanced Computer Vision
- After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. Nonetheless, you still need to decide explicitly which pixels are part of the lines and which belong to the left line or right line. **Plotting a histogram** of where the binary activations occur across the image is **one potential solution** for this.
  ```python
  import numpy as np
  import matplotlib.image as mpimg
  import matplotlib.pyplot as plt
  
  # load image. We normalised image to 0-1 since mpimg.imread load .jpy as 0-255
  img = mpimg.imread('warped_example.jpg')/255
  
  def hist(img):
      # lane lines are likely to be mostly vertical nearest to the car
      bottom_half = img[img.shape[0]//2:,:]
      
      # sum across image pixels vertically - make sure to set an `axis`.
      # the highest area of vertical lines should be larger values
      histogram = np.sum(bottom_half, axis=0)
      
      return histogram
  
  # create histogram of image binary activations
  histogram = hist(img)
  
  # visualise the resulting histogram
  plt.plot(histogram)
  ```
- **Sliding window**: With this histogram, we are adding up the pixel values along each column in the image. In our thresholded binary image, pixels are either 0 or 1, so **the two most prominent peaks in this histogram** will be good indicators of the **x-position of the base of the lane lines**. We can use that as a starting point for where to search for the lines. From that point, we can use a **sliding window**, placed around the line centres, to find and **follow the lines up to the top of the frame**.
  ![sliding_window](https://github.com/leovantoji/sdce/blob/master/images/sliding_window.png)
- We need to **split the histogram into 2 sides, one for each lane line**.
  ```python
  import numpy as np
  import cv2
  import matplotlib.pyplot as plt
  
  # take a histogram of the bottom half of the image
  histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
  
  # create an output image to draw on and visualise the result
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
  
  # find the peak of the left and right halves of the histogram
  midpoint = np.int(histogram.shape[0]//2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint
  ```
- The next step is to set up a few hyperparameters related to the sliding windows. We need to iterate across the binary activations in the image.
  ```python
  # choose the number of sliding windows
  nwindows = 9
  
  # set the width of the windows +/- margin
  margin = 100
  
  # set thee minimum number of pixels found to recenter window
  minpix = 50
  
  # set height of windows - based on nwindows above and image shape
  window_height = np.int(binary_warped.shape[0]//nwindows)
  
  # identify the x and y position of all non-zero pixels in the image
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  
  # current positions to be updated later for each window in nwindows
  leftx_current = leftx_base
  rightx_current = rightx_base
  
  # empty lists to receive left and right lane pixel indices
  left_lane_inds = []
  right_lane_inds = []
  ```
- Iterate through `nwindows` to track curvature.
  1. Loop through each window in `nwindows`.
  2. Find the boundaries of our current window. This is based on a combination of the current window's starting point (`leftx_current` and `rightx_current`), as well as the `margin`.
  3. Use `cv2.rectangle` to draw these window boundaries onto our visualisation image `out_img`.
  4. Find out which activated pixels from `nonzeroy` and `nonzerox` above actually fall into the window.
  ```python
  good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
  good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
      (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]  
  ```
  5. Append these to our lists `left_lane_inds` and `right_lane_inds`.
  6. If the number of pixels found in Step 4 is greater than `minpix`, recenter the window (i.e. `leftx_current` and `rightx_current`) based on the mean position of these pixels.
  ```python
  if len(good_left_inds) > minpix:
      leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

  if len(good_right_inds) > minpix:
      rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
  ```
- Fit a polynomial to the line after finding all pixels belonging to each line through the sliding window method.
  ```python
  # concatenate the arrays of indices
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)
  
  # extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]
  
  # fit a 2nd degree polynomial
  left_fit = np.polyfit(leftx, lefty, 2)
  right_fit = np.polyfit(rightx, righty, 2)
  
  # generate x and y values for plotting
  ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  ```
- Computing the radius of curvature of the fit.
  
  ![color-fit-lines](https://github.com/leovantoji/sdce/blob/master/images/color-fit-lines.jpg)
  
- The radius of curvature at any point *x* of the function *x = f(y)* is given as follow:
  
  ![radius_of_curvature_formula](https://github.com/leovantoji/sdce/blob/master/images/radius_of_curvature_formula.png)
  
- We need to convert from pixels to real-world metre measurements.
  ```python
  # define conversions in x and y from pixels space to meters
  ym_per_pix = 30/720 # meters per pixel in y dimension
  xm_per_pix = 3.7/700 # meters per pixel in x dimension
  
  # We'll choose the maximum y-value, corresponding to the bottom of the image
  y_eval = np.max(ploty)
    
  # implement the calculation of R_curve (radius of curvature) #####
  left_curverad = (1+(2*y_eval*left_fit_cr[0]*ym_per_pix + left_fit_cr[1])**2)**(1.5) / np.absolute(2*left_fit_cr[0])
  
  right_curverad = (1+(2*y_eval*right_fit_cr[0]*ym_per_pix + right_fit_cr[1])**2)**(1.5) / np.absolute(2*right_fit_cr[0])
  ```

## Tensorflow 1.x
- `tf.placeholder()` returns a tensor that gets its value from data passed to the `tf.session.run()` function, allowing you to set the input right before the session runs. `feed_dict` parameter in `tf.session.run()` is used to set the placeholder tensor.
  ```python
  x = tf.placeholder(tf.string)
  y = tf.placeholder(tf.int32)
  z = tf.placeholder(tf.float32)
  
  with tf.Session() as sess:
      output = sess.run(x, feed_dict={x: 'Test string', y: 123, z: 1.0})
  ```
- Basic Math functions:
  ```python
  x = tf.add(5, 2) # 7
  x = tf.subtract(10, 4) # 6
  x = tf.multiply(2, 5) # 10
  ```
- It may be necessary to convert between types to make certain operators work together.
  ```python
  # No conversion
  tf.subtract(tf.constant(2.0), tf.constant(1)) # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
  
  # After conversion
  tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1)) # 1
  ```
- `tf.Variable` class creates a tensor with an initial value that can be modified, much like a normal Python variable. This tensor stores its state in the session, so you must initialise the state of the tensor manually. The `tf.global_variables_initializer()` function is used to initialise the state of all the Variable tensors and returns an operation that will initialise all TensorFlow variables from the graph. Using the `tf.Variable` class allows us to change the weights and bias, but an initial value needs to be chosen. 
  ```python
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
      sess.run(init)
  ```
- The **weights** are often initialised as **random numbers from a normal distribution**. The `tf.truncated_normal()` function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.
  ```python
  # weight initialisation
  n_features = 120
  n_labels = 5
  weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
  ```
- The bias doesn't need to be randomised. The `tf.zeros()` function returns a tensor with all zeros.
  ```python
  n_labels = 5
  bias = tf.Variable(tf.zeros(n_labels))
  ```
- Example of `tf.nn.softmax`:
  ```python
  logit_data = [2.0, 1.0, 0.1]
  logits = tf.placeholder(tf.float32)
  softmax = tf.nn.softmax(logits)
  
  with tf.Session() as sess:
      output = sess.run(softmax, feed_dict={logits: logit_data})
      print(output)
  ```
- **Mini-batching** is a technique for **training on subsets of the dataset** instead of all the data at one time. This provides the **ability to train a model**, even if a computer **lacks the memory to store the entire dataset**. However, this technique is **computationally inefficient** since you can't calculate the loss simultaneously across all samples. 
- **Mini-batching** could be quite useful when combined with SGD. The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.
- **Mini-batching** in Tensorflow.
  ```python
  def batches(batch_size=1, features, labels):
      assert len(features) == len(labels), 'Invalid size!'
      
      output = []
      for i in range(0, len(features), batch_size):
          output.append((features[i:(i+batch_size)], labels[i:(i+batch_size)]))
      
      return output
  
  train_features = mnist.train.images
  test_features = mnist.test.images
  
  train_labels = mnist.train.labels.astype(np.float32)
  test_labels = mnist.test.labels.astype(np.float32)
  
  features = tf.placeholder(tf.float32, [None, n_input])
  labels = tf.placeholder(tf.float32, [None, n_classes])
  
  batch_size = 128
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
    
      # Train optimizer on all batches
      for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
          sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

      # Calculate accuracy for test dataset
      test_accuracy = sess.run(accuracy, feed_dict={features: test_features, labels: test_labels})

  print('Test Accuracy: {}'.format(test_accuracy))
  ```
- An **epoch** is a **single forward and backward pass** of the whole dataset. This is used to increase the accuracy of the model without requiring more data.
  ```python
  batch_size = 128
  epochs = 10
  learn_rate = 0.001

  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      # Training cycle
      for epoch_i in range(epochs):

          # Loop over all batches
          for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
              train_feed_dict = {
                  features: batch_features,
                  labels: batch_labels,
                  learning_rate: learn_rate}
              sess.run(optimizer, feed_dict=train_feed_dict)

          # Print cost and validation accuracy of an epoch
          print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

      # Calculate accuracy for test dataset
      test_accuracy = sess.run(
          accuracy,
          feed_dict={features: test_features, labels: test_labels})
  ```
- The `tf.train.Saver` class allows you to save any `tf.Variable` in your file system.
  ```python
  # file path to save data
  save_file = './model.ckpt'
  
  # Tensor variable(s) to be saved
  weights = tf.Variable(tf.truncated_normal([2, 3]))
  
  # class used to save and/or restore Tensor variable(s)
  saver = tf.train.Saver()
  
  with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      session.run(weights)
      
      # save the model
      saver.save(session, save_file)
  ```
- Load data previously saved with `tf.train.Saver` back to a new model.
  ```python
  # remove the previous weights
  tf.reset_default_graph()
  
  # Tensor variable(s) to be restored to
  weights = tf.Variable(tf.truncated_normal([2, 3]))
  
  # class used to save and/or restore Tensor variable(s)
  saver = tf.train.Saver()
  
  with tf.Session() as session:
      # load saved Tensor variable(s)
      saver.restore(session, save_file)
      session.run(weights)
  ```
- TensorFlow uses a string identifier for Tensors and Operations called `name`. If a name is not given, TensorFlow will create one automatically. TensorFlow will give the first node the name `<Type>`, and then give the name `<Type>_<number>` for the subsequent nodes. Therefore, it's important to set `name` property manually instead of letting TensorFlow does it.
  - **Erroneous** way: `InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match.`
  ```python  
  # Tensor variable(s) to be incorrectly saved
  weights = tf.Variable(tf.truncated_normal([2, 3])) # name = Variable
  biases = tf.Variable(tf.truncated_normal([3])) # name = Variable_1
  
  ############################
  # SAVED weights and biases #
  ############################
  tf.reset_default_graph()
  
  # Tensor variable(s) to be incorrectly restored to
  biases = tf.Variable(tf.truncated_normal([3])) # name = Variable
  weights = tf.Variable(tf.truncated_normal([2, 3])) # name = Variable_1
  
  saver = tf.train.Saver()
  
  with tf.Session() as session:
      # load the weights and bias - ERROR
      saver.restore(session, save_file)
  ```
  - **Correct** way:
  ```python
  # Tensor variable(s) to be correctly saved
  weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
  biases = tf.Variable(tf.truncated_normal([3]), name='bias_0')
  
  ############################
  # SAVED weights and biases #
  ############################
  
  tf.reset_default_graph()
  # Tensor variable(s) to be incorrectly restored to
  biases = tf.Variable(tf.truncated_normal([3]), name='bias_0')
  weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
  
  saver = tf.train.Saver()
  
  with tf.Session() as session:
      # load the weights and bias - SUCCESSFUL
      saver.restore(session, save_file)
  ```
- **Dropout** is a **regularisation technique** to **reduce overfitting**. The technique temporarily drops units from the network, along with all of those units incoming and outgoing connections. The `tf.nn.dropout()` function can be used to implement dropout in TensorFlow.
  - During training, good starting value for `keep_prob` is `0.5`.
  - During testing, `keep_prob` should be set to `1.0` to keep all units and maximise the power of the model.
  ```python
  keep_prob = tf.placeholder(tf.float32) # probability to keep units
  
  hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
  hidden_layer = tf.nn.relu(hidden_layer)
  hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
  
  logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
  ```
- **ConvNet**'s general idea is to **progressively squeeze** the spacial dimension while **increasing the depth** which corresponds roughly to the semantic expression of your representation.  
  ![CovNet](https://github.com/leovantoji/sdce/blob/master/images/CovNet.png)
- CNN learns to **recognise basic lines and curves**, then shapes and blobs, and then **increasingly complex objects** within the image. Finally, the CNN classifies the image by combining the larger, more complex objects.  
  ![hierarchy-diagram](https://github.com/leovantoji/sdce/blob/master/images/hierarchy-diagram.jpg)
- An example: Given input shape of `32x32x3` (HxWxD), `20` filters of shape `8x8x3` (HxWxD), a stride of `2` for both height and width, and padding size of `1`. 
  - The output will have shape of `14x14x20`.
  - Without parameter sharing, since each neuron in an output layer must connect to each neuron in the filter and a single bias neuron, the convolutional layer has `756560` (`(8x8x3 + 1) x 14x14x20`) parameters.
  - With parameter sharing, since each neuron in an output channel shares its weights with every other neuron in that channel, the convolutional layer has `3860` (`(8x8x3 + 1) x 20`) parameters.
- TensorFlow uses the following equations for `SAME` and `VALID` padding respectively.
  ```python
  # SAME padding
  out_height = ceil(float(image_height)/float(strides[1]))
  out_width = ceil(float(image_width)/float(strides[2]))
  
  # VALID padding
  out_height = ceil(float(image_height - filter_height + 1)/float(strides[1]))
  out_width = ceil(float(image_width - filter_width + 1)/float(strides[2]))
  ```
- The **benefit of the max pooling operation** is to **reduce the size of the output**, **prevent overfitting**, and **allow the neural network to focus on only the most important elements**. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining value. TensorFlow provides the `tf.nn.max_pool()` function to apply max pooling to your convolutional layers.  
  ![max-pooling](https://github.com/leovantoji/sdce/blob/master/images/max-pooling.png)
- Some **reasons not to use pooling layers** are.
  - Datasets are so big and complex, so we are more concerned about underfitting.
  - Dropout is a much better regularizer.
  - Pooling results in a loss of information.
- `1x1` convolution layer bettween a traditional convolutional layer setting provides a mini neural network running over the patch instead of a linear classifier (i.e. traditional convolutional layer). Interpersing the convolutional layers with `1x1` convolutions is a very inexpensive way to make the model deeper and have more parameters without completing changing the structure of the model. 
  ![1_by_1_conv](https://github.com/leovantoji/sdce/blob/master/images/1_by_1_conv.png)
- Inception module includes a composition of average pooling followed by `1x1`, a `1x1`, a `1x1` followed by `3x3`, and a `1x1` followed by `5x5`. There's a way to choose the parameters such that the total number of model parameters is very small. Nonetheless, the model becomes a lot more powerful than the model with only a simple convolution.
  ![inception_module](https://github.com/leovantoji/sdce/blob/master/images/inception_module.png)
- Below is an example of CNN implementation in TensorFlow 1.x. More examples can be found [here](https://github.com/aymericdamien/TensorFlow-Examples).
  ```python
  # layers of weights and biases
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

  import tensorflow as tf

  # Parameters
  learning_rate = 0.00001
  epochs = 10
  batch_size = 128

  # Number of samples to calculate validation and accuracy
  # Decrease this if you're running out of memory to calculate accuracy
  test_valid_size = 256

  # Network Parameters
  n_classes = 10  # MNIST total classes (0-9 digits)
  dropout = 0.75  # Dropout, probability to keep units

  weights = {
      'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
      'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
      'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
      'out': tf.Variable(tf.random_normal([1024, n_classes]))
  }

  biases = {
      'bc1': tf.Variable(tf.random_normal([32])),
      'bc2': tf.Variable(tf.random_normal([64])),
      'bd1': tf.Variable(tf.random_normal([1024])),
      'out': tf.Variable(tf.random_normal([n_classes]))
  }

  def conv2d(x, W, bias, strides=1, padding='SAME'):
      x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
      x = tf.nn.bias_add(x, bias)
      return tf.nn.relu(x)

  def maxpool2d(x, k=2, padding='SAME'):
      return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

  def conv_net(x, weights, biases, dropout):
      # layer 1 - 28x28x1 to 14x14x32
      conv1 = conv2d(x, weights['wc1'], biases['bc1'])
      conv1 = maxpool2d(conv1, k=2)

      # layer 2 - 14x14x32 to 7x7x64
      conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
      conv2 = maxpool2d(conv2, k=2)

      # fully connected layer - 7x7x64 to 1024
      fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
      fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
      fc1 = tf.nn.relu(fc1)
      fc1 = tf.nn.dropout(fc1, dropout)

      # output layer
      out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
      return out

  # tf graph input
  x = tf.placeholder(tf.float32, [None, 28, 28, 1])
  y = tf.placeholder(tf.float32, [None, n_classes])
  keep_prob = tf.placeholder(tf.float32)

  # model
  logits = conv_net(x, weights, biases, keep_prob)

  # loss and optimiser
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

  # accuracy
  correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # launch the graph
  with tf.Session() as session:
      session.run(tf.global_variables_initializer())

      for epoch in range(epochs):
          for batch in range(mnist.train.num_examples/batch_size):
              batch_x, batch_y = mnist.train.next_batch(batch_size)
              session.run(optimizer, feed_dict={
                  x: batch_x,
                  y: batch_y,
                  keep_prob: dropout
              })

              # calculate batch loss and accuracy
              loss = session.run(cost, feed_dict={
                  x: batch_x,
                  y: batch_y,
                  keep_prob: 1.0
              })
              valid_acc = session.run(accuracy, feed_dict={
                  x: mnist.validation.images[:test_valid_size],
                  y: mnist.validation.labels[:test_valid_size],
                  keep_prob: 1.0
              })

              print(f'Epoch {epoch+1:>2}, Batch {batch+1:>3} - Loss: {loss:>10.4f} Valid Acc: {valid_acc:.6f}')

      # calculate test accuracy
      test_acc = session.run(accuracy, feed_dict={
          x: mnist.test.images[:test_valid_size],
          y: mnist.test.labels[:test_valid_size],
          keep_prob: 1.0
      })
      print(f'Testing Accuracy: {test_acc:.2f}')
  ```
  
## Keras
- The `keras.models.Sequential` class is a **wrapper for the neural network model**. It provides common functions like `fit()`, `evaluate()` and `compile()`.
- A Keras layer is just like a neural network layer. There are fully connected layers, max pool layers, and activation layers. You can add a layer to the model using the `add()` function. Keras will **automatically infer the shape** of all layers after the first layer.
  ```python
  from keras.models import Sequential
  from keras.layers.core import Dense, Activation, Flatten, Dropout
  from keras.layers.convolutional import Conv2D
  from keras.layers.pooling import MaxPooling2D
  
  # define model architecture
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dense(5, activation='softmax'))
  
  # view model summary
  model.summary()
  
  # preprocess data
  X_normalized = np.array(X_train / 255.0 - 0.5 )

  from sklearn.preprocessing import LabelBinarizer
  label_binarizer = LabelBinarizer()
  y_one_hot = label_binarizer.fit_transform(y_train)
  
  # compile and fit the model
  model.compile('adam', 'categorical_crossentropy', ['accuracy'])
  history = model.fit(X_normalized, y_one_hot, epochs=20, validation_split=0.2)
  
  # evaluate model
  with open('small_test_traffic.p', 'rb') as f:
      data_test = pickle.load(f)

  X_test = data_test['features']
  y_test = data_test['labels']

  # preprocess data
  X_normalized_test = np.array(X_test / 255.0 - 0.5 )
  y_one_hot_test = label_binarizer.fit_transform(y_test)

  print("Testing")

  metrics = model.evaluate(X_normalized_test, y_one_hot_test)
  for metric_i in range(len(model.metrics_names)):
      metric_name = model.metrics_names[metric_i]
      metric_value = metrics[metric_i]
      print(f'{metric_name}: {metric_value}')
  ```

## Transfer learning
- GPU vs. CPU.  
  ![gpu_vs_cpu](https://github.com/leovantoji/sdce/blob/master/images/gpu_vs_cpu.png)
- [My notes about transfer learning](https://github.com/leovantoji/Machine-Learning-Engineer-Nanodegree/blob/master/Term_2_Notes.md)
- Four main cases of transfer learning:  
  ![02-guide-how-transfer-learning-v3-01](https://github.com/leovantoji/sdce/blob/master/images/02-guide-how-transfer-learning-v3-01.png)








