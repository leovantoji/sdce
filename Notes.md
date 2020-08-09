# Udacity Self-Driving Car Engineer Nanodegree
## Computer Vision Fundamentals
- From [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html) website, **Canny Edge Detection** is a popular edge detection algorithm. It was developed by John F. Canny in 1986. It is a **multi-stage algorithm**.
  - Noise reduction: Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter.
  - Finding Intensity Gradient of the image: Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction to get first derivative in horizontal direction *G<sub>x</sub>* and vertical direction *G<sub>y</sub>*. From these two images, we can find edge gradient and direction for each pixel as follows. Gradient direction is always perpendicular to edges. It is rounded to one of four angles representing vertical, horizontal and two diagonal directions.
  ![image]()
  - 
  
