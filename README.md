# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Writeup
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* data.py containing utilities that help generate and preprocess data
* model1.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model1.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used a small convolutional neural network with SELU activation function and Alpha Dropout.

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (clone.py lines 30). 

#### 3. Model parameter tuning

The model used an adam optimizer, and the initial learning rate is 1e-3. Batch size is 32. The model was trained for 15 epochs on my own dataset and then 6 more using the Udacity dataset. 

#### 4. Appropriate training data

I used a combination of self-collected data and the provided Udacity dataset. For my data collection, I tried to drive in the center of the lane with smooth steering around curves. I drove 2 laps counter-clockwise (the default direction) and 2 clockwise to help the model generalize better. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) because we know that it works for data from real cameras. However, data from the simulator is much simpler and more constrained, so the model was iteratively simplified so that we can have a small network that is much faster to train on but is still effective at controlling the car inside the simulator.

On the other hand, my general approach to data has been removing unncessary features while augmenting whenever I can. First, I cropped the top 68 pixels and the bottom 22 pixels of the images since they doesn't contain useful information (sky, trees, the front of the car etc...). After realizing that the model drove the car better at lower graphic quality, I suspected that helping the model focus on high-level representation such as road, lane lines, etc. will make it better at driving in the simulator. I can either apply a Gaussian blur or downsize the images. I chose the latter as it also allowed fast training. In addition, I augmented the data by flipping all the images horizontally and flipping the sign of the steering angle. Moreover, I used left and right images with a correction applied to the steering angle. 

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		    |     Description	        					            | Comment                                      |
|:---------------------:|:---------------------------------------------:|:--------------------------------------------:|
| Input         		    | 14x40x1 image    							                | only the S channel of the original HSV image |
| Normalization         |     							                            |                                              |
| Convolution 3x3     	| 1x1 stride, 'valid' padding, outputs 12x38x2 	|                                              |
| SELU					        |												                        | mean and variance maintained normalized      |
| Max pooling	      	  | 4x4 stride,  outputs 3x9x2 				            |                                              |
| Alpha dropout					| 0.25 keep prob											          | mean and variance maintained normalized      |
| Flatten               | outputs 54                                    |                                              |
| Fully connected		    | outputs 1      									              |                                              |
