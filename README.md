In this project I am trying to detect human from depth images(in the form of .pgm)

I am using YoloV2, Specifically TinyYolo.

Dependencies are
	-python 2
	-tensorflow
	-keras>=2.0
	-imgaug
	-tqdm
	-numpy
	-openc-cv
  
 Steps to be followed to train and test.

1) run - python gen_anchors.py 
	-This will generate the anchors for the data we have. It takes the bounding boxes from labels.txt file. Anchors are the general bounding box ratios of width to height in our data.
2) run - python split_data.py
	-This will spilt our data to training and validation set.
3) run - python yolo.py
	-This is the script I wrote for training and also testing. I made it in a single file mainly because it will be easier to debug.
	-Once training is done commentout model.fit_generator
	-To test uncomment the part below #-----TEST-----#
  
  
For training your own dataset, The hyperparameters to tweek with are.

-epochs - Number of training cycle
-learning rate ('lr' in optimiizer)
-custom loss function. This i have used the same loss function in this link https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation 
-If augumentaion is needed in training batch put jitter=True
-Add or Remove layers in the Tiny Yolo network.
-We can also change the number of filters in the layers. 
-Adjust the 4 scale values. The scale values is used to determine how much to penalize prediction of confidence of object predictors.
-change the number of warmup batches(between 0-5). Warmup training is used to get adjusted with the inputs anchors and sizes initially for the network.
-Change the batch size. Its always better to increase it according to the hardware available.
