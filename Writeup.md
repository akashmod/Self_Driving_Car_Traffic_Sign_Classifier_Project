
#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/plot.jpg "Visualization"
[image2]: ./examples/col.jpg "Colour image"
[image3]: ./examples/bw.jpg "Grayscale image"
[image4]: ./examples/test1.jpg "Traffic Sign 1"
[image5]: ./examples/test5.jpg "Traffic Sign 2"
[image6]: ./examples/test3.jpg "Traffic Sign 3"
[image7]: ./examples/test9.jpg "Traffic Sign 4"
[image8]: ./examples/test11.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Answer : I used the python library numpy to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Answer : Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of images in the training data over the different classes.
![alt text][image1]

The figure helps us visualize the number of images in each class in the training set,i.e. from 0 to 43.


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Answer : As a first step, I decided to convert the images to grayscale because the color content in the image is not the chief contributor towards the classification of the images. The grayscale images would help the learning process by avoiding the colour characteristics from determining the weights. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data because for the learning process the scale of all features should be similar. Hence I had subtracted 128 and then divided by 128 for each pixel. This gives us the features values scaled between -1 and 1. I had also added an additional axis for the matrix files to make the shape acceptable to the convolution algorithm. The shape of the final image is 32x32x1x1


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Answer : My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Rectified Linear Unit Activation				|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	     			|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU          		| Rectified Linear Unit Activation				|
| Max pooling			| 2x2 stride, outputs 5X5x16        			|
| Fully Connected Layer | The input is flattened, output is 120x1		|
| RELU              	| Rectified Linear Unit Activation				|
| Fully Connected Layer	| Outputs 84x1  								|
| RELU					| Rectified Linear Unit Activation				|
| Dropout				| To provide a dropout of 0.5 to avoid overfitting|
| Fully Connected Layer	| outputs 43x1, the logits						|
 
The first 2 layers involved 5x5 convolution layers to extract the features from the images, with alternate max pooling and relu activation. The three fully connected layers follow the convolution layers with a dropout layer of 0.5 keep probability to reduce the overfitting problem.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Answer : To train the model, I used an Adam Optimizer with the batch size of 128, number of epochs 20 and the learning rate 0.001. The probability was calculated using softmax of the logits and the cross-entropy loss of the model. This error was then minimized to train the model.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Answer : 
My final model results were:
training set accuracy of 99%
validation set accuracy of 95% 
test set accuracy of 100%

First, I chose the multi-scale architecture as described in the paper mentioned in the description of the project. The multiscale architecture involved sending the outputs of each of the layers to the classifier. This ensures the use of low level features in the classification of the images. However, I realized this only increased the training accuracy of the model rather than the validation accuracy of the model. The increase of the validation accuracy would require the resolution of the overfitting problem of the model. I used a dropout layer at the end of the model, i.e. at the layer before the classifier to avoid losing of information by incorporating the dropout layer earlier in the model. I had also tried using two dropout layers in the model but that only led to underfitting problem and hence the idea was dropped.
The dropout layer deactivated half of the elements in the final logits before the classifier to remove the dependence of the model on a few elements in the model. The training of the model using dropout takes higher epochs. The validation accuracy is seen to rise slowly throughout the iterations. Care must be taken however to remove the dropout layer when evaluating the accuracy.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Answer : 
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of the noise in the background that interferes with the image. It is difficult for a human to understand the image let alone the program. The second image is difficult to classify because of the pole and the white board below the image which can interfere with the actual image. The third image is difficult to classify because of the involved background which can interfere with the actual image classification. The fourth image is difficult to classify again because of a bright background. The fifth image can be confused with a stop sign by the program.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Answer : 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30   		| Speed Limit 30   								| 
| Turn Right Ahead  	| Turn Right Ahead								|
| Keep Right			| Keep Right									|
| Ahead Only      		| Ahead Only					 				|
| Speed Limit 100		| Speed Limit 100      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Speed Limit 30 kmph 					        | 
| 0     				| Speed Limit 50 kmph						    |
| 0					    | Speed Limit 80 kmph						    |
| 0	      		    	| Speed Limit 20 kmph 				            |
| 0			    	    | Speed Limit 70 kmph				            |

By image probabilities for the first image, we can see that the program is very sure that the sign is speed limit 30. The other signs are the other speed limits, as expected.

For the second image, the top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Turn Right Ahead 					            | 
| 0     				| Keep Left						                |
| 0					    | Stop						                    |
| 0	      		    	| Priority Road 				                |
| 0			    	    | Double Curve 				                    |
 
Once again, the program is pretty sure about the sign being Turn Right Ahead. 

For the third image, we see that the program has a 100% probability that it is Keep Right.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Keep Right 					                | 
| 0     				| Speed Limit 60 kmph			                |
| 0					    | Turn Left Ahead				                |
| 0	      		    	| Speed Limit 80 kmph 			                |
| 0			    	    | Dangerous Curve to the Right                  |

For the fourth image, we see that the program has a 100% probability that it is ahead only. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Ahead Only 					                | 
| 0     				| No Passing             		                |
| 0					    | Go Straight or Right			                |
| 0	      		    	| Turn Left Ahead 			                    |
| 0			    	    | Yield                                         |

For the fifth image, we see that the program has a 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        			| Speed Limit 100 kmph 			                | 
| 0     				| Speed Limit 80 kmph			                |
| 0					    | Speed Limit 120 kmph			                |
| 0	      		    	| Speed Limit 50 kmph 			                |
| 0			    	    | Roundabout mandatory                          |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?





```python

```
