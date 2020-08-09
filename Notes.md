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
  















