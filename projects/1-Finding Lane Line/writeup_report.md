# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./test_images_output/whiteCarLaneSwitch_output.jpg "whiteCarLaneSwitch_output"


### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consists of the following steps:
1. Convert colorful image to grayscale.
2. Filter out noises in the image using a Gaussian Noise kernel.
3. Detect edges in the image using Canny Edge Detection algorithm.
4. Identify a region of interest in the image and keep the detected edges in this region. The rest of the edges outside the region are discarded.
5. Detect lines using Hough transformation.
6. Based on lines detected from step 5, find a single average line for each of the left and right lane line. Extrapolate the left and right lane lines such that they cut the region of interest identified in step 4.
7. Draw the extrapolated lines on the image.

In order to draw a single line on the left and right lanes, slopes for each line segments identified by Hough transformation step are calculated. Negative and Positive slopes and y intercepts are separated as they correspond to the left and right lane line respectively. A pair of median slope and y intercept are then constructed for both the left and right lane line. Median was used instead of mean because I want to eliminate of certain outlier line segments. The top and bottom of the lane were identified as the minimum and maximum y values of the various line segments which are fed into the function through parameter `lines`. Having these additional information allows the left and right lane lines to be extrapolated to the top and bottom of the lane.

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

A few shortcomings with my current pipeline:
- Vertices for region of interest were originally static, which means if the car is unstable resulting in the camera changing from its mounted position, the lane detection pipeline will probably fail. 
- The left and right lane slope and y intercept were originally estimated using mean. Nonetheless, I realised using mean can suffer from irrelevant line segments (outliers) detected in the Hough transformation step. 
- GaussianBlur filter may also blur the edges detected. 
- The current pipeline doesn't deal with other potential noises: shadows, dirts on the camera, change of weather, etc. All of these can hinder the working of the current pipeline.


### 3. Suggest possible improvements to your pipeline

A few improvements to my current pipeline:
- Dynamically identify vertices for the region of interest.
- Use median instead of mean to identify the left and right lane slope and y intercept. As mentioned earlier, using mean suffers from a lot of noises, while median generally can avoid the effect of outlier line segments.
- Use different filtering (e.g. BilateralFilter) which is both effective at noise removal and preserving edges.
- Add additional image processing step to deal with change of weather condition.
