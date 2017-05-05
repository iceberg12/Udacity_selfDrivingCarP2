# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_report/1.png "Visualization"
[image2]: ./images_report/2.png "Grayscaling"
[image3]: ./images_report/3.png "Random Noise"
[image4]: ./images_report/4.png "Traffic Sign 1"
[image5]: ./images_report/5.png "Traffic Sign 2"
[image6]: ./images_report/6.png "Traffic Sign 3"
[image7]: ./images_report/7.png "Traffic Sign 4"
[image8]: ./images_report/8.png "Traffic Sign 5"
[image9]: ./images_report/9.png "Visualization"
[image10]: ./images_report/10.png "Grayscaling"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/iceberg12/Udacity_selfDrivingCarP2/blob/master/Traffic_Sign_Classifier-Copy1.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

We can see that the classes are imbalanced. Thus, in the augmentation step later, I will augment minor classes more than major classes.

Also, I plot out an example for each class, with their labels obtained from the reference signnames.csv.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### 1a. I decided to convert the images to grayscale because the traffic design does not depend on color. Thus, by using grayscale we can reduce the model size a little. As a first step, I use the ideas in the paper Traffic Sign Recognition with Multi-Scale Convolutional Networks by Pierre Sermanet and Yann LeCun. It is about a transformation based on global and then local contrast normalization, so I implemented using skimage and scipy.ndimage. Furthermore, I compare transformation in YUV and RGB space and realize RGB gives a better edge performance than YUV discussed in the paper, so I choose tranformation in RGB. After that, grayscale conversion follows with the weight [0.2126, 0.7152, 0.0722]. 

Here is an example of a traffic sign image before and after transforming and grayscaling.

![alt text][image3]

As a last step, I normalized the image data to range [-1, 1] because it helps with the weight training of our neural network.

##### 1b. I also generated additional data in order to create more realistic data to help the model recognizes the traffic signs under different conditions.

To add more data to the the data set, I used the following techniques:
* Rotatation [-15, 15] degrees
* Translation [-4, 4] pixels in both x, y dimensions
* Zoom [80%, 120%]

This is done in a random augmentation for each grayscale image. Furthermore, depending on the population of each class of traffic signs, I augment it proportionally. 

The augmented data set has a more balanced distribution

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5, RELU	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Dropout				| 0.9											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5, RELU	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Dropout				| 0.7											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Convolution 3x3, RELU | 1x1 stride, valid padding, outputs 3x3x32 	|
| Dropout				| 0.7											|
| Max pooling	      	| 1x1 stride,  outputs 2x2x32   				|
| Flatten + Combine    	| Combine layers 5x5x16 and 2x2x32 to get 528x1 |
| Fully connected		| 528x120      									|
| Dropout				| 0.5											|
| Fully connected		| 120x84      									|
| Dropout				| 0.5											|
| Softmax				| 84x43        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer to optimize the learning rate automatically, batch size 128, number of epochs 30 and values for learning rate is 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.748
* validation set accuracy of 0.953 
* test set accuracy of 0.936

The result was mainly boosted by image processing, and further improved by model tuning.
* The first architecture that was tried was LeNet because of its solid and simple structure 
* Some problems with the initial architecture are overfitting of training set, and then low accuracy (~0.89).
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* The tuned parameters are dropout rate and learning rate. Dropout rates were adjusted based on overfitting of training set i.e. the gap between training and validation accuracy. Learning rate was adjusted to be smaller when the training accuracy kept fluctuating and did not increase. 
* Convolution layers typically use the same filter weights and run across images so it makes training faster and easier. Dropout technique further makes the network to learn with redundancy because of random dropout of neurons, thus makes the performance more generalized and robust. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Turn left ahead     	                | Turn left ahead								| 
| General caution     	                | General caution								|
| 30 km/h				                | Keep right									|
| No vehicles      		                | Go straight or right			 				|
| Right of way at the next intersection	| Beware of ice/snow   							|

![alt text][image6]

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. The last picture, Right of way at the next intersection, looks a bit similar to Beware of ice/snow. This result does not agree well with the test accuracy, which is 0.936.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Turn left ahead sign (probability of 0.6), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .25         			| Turn left ahead								| 
| .66     				| General caution 								|
| .16					| Keep right									|
| .20	      			| Go straight or right					 		|
| .14				    | Beware of ice/snow                			|

![alt text][image7]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Convolution layer 1

![alt text][image8]

Convolution layer 2

![alt text][image9]

Convolution layer 3

![alt text][image10]

I applied dropout for convolution layers, believing that it still helps with regularization although the number of neurons is not large compared to fully connected layers. From the visualization, it can be seen that for validation data the End of speed limits / End of no passing signs play a large portion because from Convolution layer 1 we see a lot of diagonal line features. For convolution 2 and 3, the sizes are small so we can't grasp a meaningful sense out of the images. One way to improve is keeping the outputs of these layers at high resolution by applying SAME padding, but note that max pooling still reduces the feature map size.
