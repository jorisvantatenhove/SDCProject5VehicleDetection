# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/car1.png "Car 1"
[image2]: ./examples/car2.png "Car 2"
[image3]: ./examples/car3.png "Car 3"
[image4]: ./examples/car4.png "Car 4"
[image5]: ./examples/noncar1.png "Noncar 1"
[image6]: ./examples/noncar2.png "Noncar 2"
[image7]: ./examples/hog_still_frame.png "HOG Procedure"
[image8]: ./examples/windows.png "Windows"
[image9]: ./examples/frame5_boxes.png "Boxes"


The aim of the following project is finding all vehicles in every frame in an input video. We do this by splitting up the frame into smaller squares, and then decide per square whether we recognize car or not. We train a classifier using a heatmap of gradient (HOG) approach.

## Data

We have a dataset containing about 9000 64x64 images of cars, and also of about 9000 64x64 images of non-car things that you might end up seeing on the road: lane lines, trees, guardrails and so on. Below are a couple of examples:

### Cars
![Car1][image1]
![Car2][image2]
![Car3][image3]
![Car4][image4]

### Noncars
![Noncar1][image5]
![Noncar2][image6]

## Classifier

We can use these 18,000 images to train our classifier. As suggested, we've chosen a LinearSVC (see svc_utility.py) with only HOG features (see hog_utility.py). After tweaking the parameters a bit, we already achieved an accuracy of above 98%, without using spatial or color features, so we did not deem this necessary.

We did this by switching to the YUV color channel, and taking all three channels into consideration. We split the 64x64 image into cells of 8x8, and determine the gradient direction for each of these cells. The gradient snaps to one of 16 directions. We normalize a group of 2x2 cells. We apply this for every color channel (this car is actually taken from a screenshot from the video!):

![HOG Procedure][image7]

All these gradients together for the basis on which we trained our classifier.

## Sliding window 

For a given frame, we now want to divide the image into squares in which we can expect to find cars, and identify them using the classifier we've trained.

We look for the following squares:

![Windows][image8]

Given all windows, we now want to take out the false positives. To do this, we implement a threshold function using a heatmap. We overlap all windows that return a positive classification, and then if a pixel has enough overlap, we interpret this as part of a vehicle.
All the distinct blobs of pixels are then the vehicles. How this process work is displayed on its performance on test image 5, where there is actually some overlap in the initial windows, but that no longer exists after thresholding:

![Boxes][image9]

## End result and discussion

The end result can be found in project_output.mp4.

This result can be improved by further getting rid of the false negatives, probably by taking history. Something like a weighted average of the heatmap of the past couple of frames could really help us getting rid of the negatives in the middle of the road.
Alternatively, we could train our classifier better by providing more noncar data of empty highway.

Moreover, we identify some vehicles in the incoming traffic. When driving on a highway, this is irrelevant information and will only pollute the results. When driving on a road without separation between the two driving directions however, this should be improved because then this is really important!

Lastly, we should also train for motor bikes! We would not recognize them with the current classifier, which could potentially be very dangerous!!