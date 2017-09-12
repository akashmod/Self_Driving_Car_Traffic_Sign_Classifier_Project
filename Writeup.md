
# **Traffic Sign Recognition** 

## Writeup 

---

**Build a Traffic Sign Recognition Project**

The steps involved in the project are:
* Loading the data set (see below for links to the project data set)
* Exploring, summarizing and visualizing the data set
* Designing, training and testing a model architecture
* Using the model to make predictions on new images
* Analyzing the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./examples/plot.jpg "Visualization"
[image2]: ./examples/col.jpg "Colour image"
[image3]: ./examples/bw.jpg "Grayscale image"
[image4]: ./examples/test1.jpg "Traffic Sign 1"
[image5]: ./examples/test5.jpg "Traffic Sign 2"
[image6]: ./examples/test3.jpg "Traffic Sign 3"
[image7]: ./examples/test9.jpg "Traffic Sign 4"
[image8]: ./examples/test11.jpg "Traffic Sign 5"

### The [rubric points](https://review.udacity.com/#!/rubrics/481/view) described for the project were met with satisfaction and submitted to Udacity and graded by them.  

#### This writeup file details how I addressed each of the rubric points and the general structure of the python code written to classify traffic signs. 

### Data Set Summary & Exploration

Below is the summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

### Exploratory Visualisation of the dataset
Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of images in the training data over the different classes.
![alt text][image1]

The figure helps us visualize the number of images in each class in the training set,i.e. from 0 to 43.


### Design and Test a Model Architecture

#### Preprocessing of the data

As a first step, I decided to convert the images to grayscale because the color content in the image is not the chief contributor towards the classification of the images. The grayscale images would help the learning process by avoiding the colour characteristics from determining the weights. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data because for the learning process the scale of all features should be similar. Hence I had subtracted 128 and then divided by 128 for each pixel. This gives us the features values scaled between -1 and 1. I had also added an additional axis for the matrix files to make the shape acceptable to the convolution algorithm. The shape of the final image is 32x32x1x1


#### The Final Model Architecture 
My final model consisted of the following layers:

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

#### Training of the Model

To train the model, I used an Adam Optimizer with the batch size of 128 and number of epochs 20. Since it was as Adam Optimizer, there was separate parameter for the learning rate required. The probability was calculated using softmax of the logits and the cross-entropy loss of the model. This error was then minimized to train the model.

#### The Steps Performed to improve the model accuracy beyond 93%.
 
My final model results were:
training set accuracy of 99%
validation set accuracy of 95% 
test set accuracy of 100%

First, I chose the multi-scale architecture as described in the paper mentioned in the description of the project. The multiscale architecture involved sending the outputs of each of the layers to the classifier. This ensures the use of low level features in the classification of the images. However, I realized this only increased the training accuracy of the model rather than the validation accuracy of the model. The increase of the validation accuracy would require the resolution of the overfitting problem of the model. I used a dropout layer at the end of the model, i.e. at the layer before the classifier to avoid losing of information by incorporating the dropout layer earlier in the model. I had also tried using two dropout layers in the model but that only led to underfitting problem and hence the idea was dropped.
The dropout layer deactivated half of the elements in the final logits before the classifier to remove the dependence of the model on a few elements in the model. The training of the model using dropout takes higher epochs. The validation accuracy is seen to rise slowly throughout the iterations. Care must be taken however to remove the dropout layer when evaluating the accuracy.
 

### Testing the Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because of the noise in the background that interferes with the image. It is difficult for a human to understand the image let alone the program. The second image is difficult to classify because of the pole and the white board below the image which can interfere with the actual image. The third image is difficult to classify because of the involved background which can interfere with the actual image classification. The fourth image is difficult to classify again because of a bright background. The fifth image can be confused with a stop sign by the program.

#### Each of the images were now classified by the model to the sign type.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30   		| Speed Limit 30   								| 
| Turn Right Ahead  	| Turn Right Ahead								|
| Keep Right			| Keep Right									|
| Ahead Only      		| Ahead Only					 				|
| Speed Limit 100		| Speed Limit 100      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### Top 5 softmax probabilities of each of the signs

Next I have included top 5 softmax probabilities of each of the five images classified above to see with what confidence does the classifier classify them.
The top five soft max probabilities for the first image were

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




