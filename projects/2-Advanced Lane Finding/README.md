## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, I created a software pipeline to identify the lane boundaries in a video. The major steps are listed below.
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/camera_cal.jpg "Camera Calibration"
[image1]: ./output_images/Undistorted.jpg "Undistorted"
[image2]: ./output_images/Undistorted2.jpg "Undistorted2"
[image3]: ./output_images/binary_combo.jpg "Binary Example"
[image4]: ./output_images/birds_eye.jpg "Warp Example"
[image5]: ./output_images/colour_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/final_output.jpg "Output"
[image7]: ./output_images/R_channel.jpg "R Channel"
[image8]: ./output_images/L_channel.jpg "L Channel"
[image9]: ./output_images/Mag_Dir_Grad.jpg "Magnitude & Direction of Gradients"
[image10]: ./output_images/lane_detected.png "Lane Detected"
[image11]: ./output_images/warped.jpg "Processed Warp"
[video]: ./output_videos/project_video_output.gif "Video"

### Step 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
The camera was calibrated using a set of images of a chessboard. For each image, corners were detected and drawn on the image as shown below.

![Camera Calibration][image0]

### Step 2: Apply a distortion correction to raw images
The calibration matrix and distortion coefficients were applied on raw images to rectify any distortion caused by the camera.

![Undistorted][image1]
![Undistorted2][image2]

### Step 3: Create a thresholded binary image using colour transform, gradients, etc.
A combination of thresholding in RGB and HLS colour space, gradient in both horizontal and vertical direction, and direction of the gradient was deployed to transform the original image to a binary image. The goal is to remove noises and retain lane lines from the original image. 
* RGB threshold in Red colour channel could mostly capture both left and right lane lines. Red colour channel was chosen such that the yellow colour lane could be retained after the thresholding process.
![R Channel][image7]

* HLS threshold in L channel mostly captured the right lane.
![L Channel][image8]

* While thresholding in RGB and HLS colour space could mostly do the job, the remaining 2 gradient calculations managed to fill in some additional detail that were filtered out by the previous 2 transformations.
![Mag Dir Grad][image9]

* Final output.
![Binary Example][image3]

The thresholding steps can be found in the `pipeline()` function available in the `Test pipeline on Videos` section of the `Advanced_Lane_Finding.ipynb` notebook. The code below extracted from `pipeline()` shows the exact threshold parameters applied.

```python
birdEyeProcessed = binary_threshold(birdEye, 
                                    magIn=(5, (20, 90)), 
                                    dirIn=(15, (0.3, 0.7)),
                                    rgbInR=(225, 255), 
                                    hlsInL=(200, 255))
```

### Step 4: Apply a perspective transform to rectify binary image ("birds-eye view")
I performed a perspective transform on raw input images in order to view the lane lines from above. The code for my perspective transform includes a function called `warp()`, which appears in section under the same name as the step in the `Advanced_Lane_Finding.ipynb` notebook. The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. The `warp()` function returns the warped image, the perspective transform matrix and the inverse perspective transform matrix.

The source and destination points are hard-coded after careful examination of the sample images.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 568, 468      | 200, 0        | 
| 715, 468      | 1000, 0       |
| 1040, 680     | 1000, 680     |
| 270, 680      | 200, 680      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warp Example][image4]

### Step 5: Detect lane pixels and fit to find the lane boundary
After the last 4 steps, I have obtained a binary image from top-down view where lane lines stood out clearly. This step shows the process to identify the pixels belonging to the left and right lane line.

The binary thresholded warped image at the start of step 5.
![Processed Warp][image11]

The lane line pixels were detected by:
* Plotting a histogram of the lower half of the binary image. Only the lower half of the image is needed as we can expect the lane lines to be mostly vertical and near to the car. 
* Dividing the histogram in half and finding the maximum values of the histogram in both halves allow us to identify the respective location of the left and right lane line.
* A sliding window, which moves upward from the highest peaks identified in the previous step, will determine where the lane lines go.
* Upon finding all the pixels belonging to the lane lines, we'll fit a second degree polynomial to the line. 

The detected lines after step 5.
![Lane detected][image10]

### Step 6: Determine the curvature of the lane and vehicle position with respect to center
The quadratic equation to approximate the lane line is in the form: f(y) = Ay<sup>2</sup> + By + C. Thus, the radius of the curvature was computed using the following formula: R<sub>curve</sub> = (1 + (2Ay + B)<sup>2</sup>)<sup>1.5</sup> / 2|A|. In order to calculate the curvature of the lane in metres, a pixel-to-metre conversion was applied, and the polinomial was refit again.

The vehicle position with respect to the centre of the lane was approximated as the difference between the centre of the image, which represents the position of the car, and the average of the lane pixels closest to the car.

I implemented a function called `measure_stats()` to perform these 2 calculations.

### Step 7 & 8: Warp the detected lane boundaries back onto the original image. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

The detected lane boundaries were warped back to the original (undistorted) image using the inverse perspective transform matrix obtained in step 4. The space between the 2 polynomials representing the 2 lanes was filled. In addition, the numerical estimation of the lane curvature and vehicle position were also displayed in the output image. Below is an example of the output.
![Output][image6]

### Pipeline (video)
The whole pipeline consisting of the previous steps was applied to a video. As the lane lines might not be too different from frame to frame, the information of the previously detected lane lines was stored, and used as a guide to detect new lines. The new lines would be searched from a position to the left and right of the previously detected lines. The green-shaded region in the image below shows the region used to search for the new lines in the next frame.
![Fit Visual][image5]

In addition, if the newly detected lanes diverge too far from previously detected lanes, the newly detected lanes will be discarded. The lane for each new frame is the average of the last 3 detected lines.

Below is the output.
[video output][video].

### Discussion
#### Limitations
- The hard-coded source and destination points used in `warp()` function pose a serious challenge if there is significant change in the image feed. Bumpy roads which unstablise the camera could be a cause for this scenario.
- The process to generate a binary image isn't robust enough to remove all noises from the original image. Too many shadows in the image is a known issue of the current pipeline.

#### Area of Improvement
- The current pipeline fails miserably if there are sharp turns or lots of unwanted markings in the road. I believe this failure is due to the weakness of the binary image generation process. Currently, the hyperparameters were chosen manually, while I believe they could be fine-tuned automatically.
- Additionally, I want to look at areas to automatically approximate the source and destination points for perspective transform.
