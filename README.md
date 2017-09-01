# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: model.png "Model Visualization"
[image2]: ./examples/center_1.jpg "Center Camera Recovery 1"
[image3]: ./examples/center_2.jpg "Center Camera Recovery 2"
[image4]: ./examples/center_3.jpg "Center Camera Recovery 3"
[image5]: ./examples/not_flipped.jpg "Actual Image"
[image6]: ./examples/flipped.jpg "Flipped Image"
[data_cleanup]: ./examples/data_cleanup.png "Data Cleanup Process"
[brightness]: ./examples/brightness.png "Brightness Augmentation"
[verticalshift]: ./examples/shifting.png "Random Veritical Shift"
[prediction1]: ./examples/prediction.png "Predicted Image"
[prediction2]: ./examples/prediction2.png "Predicted Image"
[prediction3]: ./examples/prediction3.png "Predicted Image"

---
**Files Submitted & Code Quality**

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* CarND-Behaviour-Cloning-Keras.ipynb a notebook to briefly explain the process

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have built this model using the popular NVIDIA architecture.

It consists of 5 convolution neural networks followed by 3 fully connected networks.

Model contains a normalization and mean centered layer.  It also has cropping layer to remove unneeded portions of the images(top and bottom).

To prevent overfitting, I have used L2 regularization technique as this is well known approach to address data uniformity issues like zero angle bias in the current data set.

I have not used any dropout or softmax activation to the final output, as this is a single value output predictions.

I have used the mse loss function along with adam optimizer to train the model.

To overcome issues with memory during training and validation, I've made use of generators for both training and validation data.

####2. Attempts to reduce overfitting in the model

Before I trained the model, i have made sure to capture the weights through checkpoint callbacks.

During the training process, I first trained the model with track data with 2 straight drives.

After the training, I noticed the car is having troubles near the corners and over the bridge.

I have captured more driving data around the bridge along with driving away from the edges.

Now, with the help of checkpoints i have used the earlier trained weights and trained the model with new training data.

I followed the same approach for resolving few more issue around corners and shades.

####3. Model parameter tuning

I have used a correction factor of 0.25 to adjust steering angles from left and right.

I have used a learning rate of 1e-5.

I have used 5 epochs to train and validate the data.

I have used 128 items for each epoch.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I also drove the car in the opposite direction to prevent model from being predictable.

I drove more around the bridge to handle issues like staying close to the center, recover from the ends etc.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with few laps of data while keeping close to center. Now, use the model to observe the car driving behavior in autonomous mode. Based on results, i have recorded extra data to record car recovering from the edges towards the center. I also recorded additional data around the corners.

Due to the use of checkpoints, i was able to use the trained weights from the previous trained data and then improve the model with the new data.

I honestly felt the NVIDIA architecture that was presented during course is pretty self sufficient to perform this task.

I tried to add dropouts layers, but noticed the model is underfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and then data for recovery and corners. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer away from the sides. These images show what a recovery looks like starting from left to center:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would prevent the car driving close to the side ways. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

After the collection process, I had more than 20000 data points.

From the data records, I noticed the data is seriosly biased towards zero angle measurements which might cause the model to drive car in straight paths only. To overcome this, I have reduced the zero angle records count to 70% and also flipped images with angles greater than 0.3. This reduced the data distribution difference a little.

![alt text][data_cleanup]

To generalize the model for other driving lanes and light conditions, I have random brightness to images as part of data augmentation step.

![alt text][brightness]

I have also included a random veritcal shift to the images to handle driving over slope scenarios.

![alt text][verticalshift]

I then used to generators and preprocessing layers like normalization and cropping layers to optimize the data set.

I finally randomly shuffled the data set and put 2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Test the model

To test the model, I have randomly picked a record from dataset and used model to predict the measurement. I then used the prediction and actual measurement to draw lines on the actual images.(Green line for actual angle and blue line for predicted measurement.)

![alt text][prediction1]

![alt text][prediction2]

![alt text][prediction3]
