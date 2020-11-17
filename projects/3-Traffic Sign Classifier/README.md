# **Traffic Sign Recognition** 

### Writeup
---

The major steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./test_images/class_distribution.png "EDA"
[image2]: ./test_images/grayscale_normalisation.png "Processed"
[image3]: ./test_images/random_images.png "Random Images"
[image4]: ./test_images/bumpy_road.jpg "Traffic Sign 1"
[image5]: ./test_images/german-100.jpg "Traffic Sign 2"
[image6]: ./test_images/german-road-sign-caution-shaded-road-J2MTPA.jpg "Traffic Sign 3"
[image7]: ./test_images/german-traffic-sign-caution-roadworks-71151565.jpg "Traffic Sign 4"
[image8]: ./test_images/round_about.jpg "Traffic Sign 5"
[image9]: ./test_images/traffic_sign_softmax.png "Traffic Sign Softmax"

### Data Set Summary & Exploration

#### 1. Basic Statistics

I used the pandas library to calculate summary statistics of the traffic signs data set:

* Training set consists of 34,799 images.
* Validation set consists of 4,410 images.
* Testing set consists of 12,630 images.
* The shape of a traffic sign image is 32x32x3 (Height x Width x Depth).
* The number of unique classes/labels in the data set is 43.

Random images and respective grayscaled images in the dataset are shown below.

![Random Images][image3]

#### 2. Exploratory Visualisation of Dataset

This is a bar chart showing the number of images for each class in each of the training, validation, and testing dataset. The classes are unevenly distributed and this could affect the performance of the classification model. 

* Least represented class - Speed limit (20km/h) - 180 images.
* Most represented class - Speed limit (50km/h) - 2,010 images.

![EDA][image1]

### Design and Test a Model Architecture

#### 1. Image Preprocessing

* Images are first converted to grayscale as I noticed the lighting conditions vary greatly from one image to another. This problem appears to be mitigated after the grayscaling process. In addition, it is stated in this [article](https://ieeexplore.ieee.org/document/7562656) that grayscale images resulted in higher accuracy classification than RGB images.
* 2 different normalisation methods were tried. The goal is to ensure that the data has mean zero and equal variance.
    * Simple: `(pixel - 128)/128`.
    * Complex: `(pixel - global_mean)/global_standard_deviation`.

![Processed][image2]

#### 2. Final Model Architecture

My final model architecture is similar to that of [Lenet-5](http://yann.lecun.com/exdb/lenet/). This is a simple model that is known to perform well on the MNIST dataset. I modified the original architecture slightly by adding 3 additional Dropout layers to prevent overfitting.

|Layer|Description| 
|:-:|:-:| 
|Input|32x32x1 RGB image| 
|Convolution 5x5|1x1 stride, valid padding, outputs 28x28x6|
|Max pooling|2x2 stride, valid padding, outputs 14x14x6|
|RELU|Activation|
|Convolution 5x5|1x1 stride, valid padding, outputs 10x10x16|
|Max pooling|2x2 stride, valid padding, outputs 5x5x16|
|RELU|Activation|
|Flatten|outputs 400|
|Dropout|Dropout Rate: 50%|
|Fully connected|outputs 120|
|Dropout|Dropout Rate: 50%|
|Fully connected|outputs 84|
|Dropout|Dropout Rate: 50%|
|Fully connected|outputs 43 - Number of classes|

#### 3. Model Training

To train the model, I used the Adam Optimizer and the following hyperparameters:
* The number of epochs is 100. 
* Batch size of 128.
* Learning rate of 0.0005.
* Dropout rate of 0.5.

#### 4. Approach

I chose an iterative approach and obtained the following final results. The difference between training, validation and testing accuracy is small (< 5%) indicating that overfitting is not a major concern of the final model.
* Training set accuracy of 99.2%.
* Validation set accuracy of 96.6%.
* Test set accuracy of 94.6%.

The first architecture I tried was LeNet model without Dropout layers. Originally I tried having inputs as RGB images but soon realised that the test accuracy was not as good as the result obtained from having inputs as grayscale images. Nonetheless, both methods only achieved ~85% test accuracy, and overfitting was clearly observed from the big gap between the training accuracy and the validation accuracy. 

In order to improve the model, I tried to add a few additional dropout layers to prevent overfitting. In addition, the test accuracy was improved greatly by increasing the number of times the network passing through the entire training dataset. Furthermore, as stated above, I tried 2 data normalisation approaches and eventually chose the simple method as that approach yielded better test accuracy. 

### Test a Model on New Images

#### 1. Five random German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![General caution][image6] ![Bumpy road][image4] ![Speed limit (100 km/h)][image5] ![Roundabout][image8] ![Road work][image7]

All images are of good quality and bright in colours. This could prove to be a challenge since the training dataset contains lots of dark images. Nonetheless, the grayscale transformation was there to address this issue of varying lighting conditions.

#### 2. Performance

Here are the results of the prediction:

|Image|Prediction|
|:-:|:-:|
|General caution|General caution|
|Bumpy road|Bumpy road|
|100 km/h|50 km/h|
|Roundabout mandatory|Roundabout mandatory|
|Road work|Road work|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. I did not expect the model to fail on the `Speed limit (100 km/h)` image. Surprisingly, the top guesses the model had for the image were all different speed limit signs, but none of the guesses were `Speed limit (100 km/h)`.

#### 3. Prediction Softmax Probabilities

Results can be seen below.

![Softmax probabilities][image9]
