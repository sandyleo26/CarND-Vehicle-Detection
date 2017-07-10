---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_window.png
[image5]: ./examples/heatmap.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 0. Submitted files
    * writeup.md
    * P5_Vehicle_Detection.ipynb
    * project_video_processed.mp4

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code is in the 4th code cell of the notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled the following. I found using the following parameters along with color histogram and spatial binning could get me very good test accuracy.

    color_space = 'YCrCb'
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

Here is the parameters I use for spatial binning and color histogram.

    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all 3 channel3 HOG, spatial binning and color histogram features concatenated together. The feature vector length is `6108`. The 7th code cell is where the training happens.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at the bottom half of the images using a slide window approach.

First HOG feature is calculated on the entire bottom half. Then a `64x64` window (with 8 cells and 8 pix per cell) is used with a stride of `2` cells. For each window, the hog feature can be sliced using previously calculated HOG feature.

I use a scale of `1.5` which works the best in terms of rejecting false positives and improving accuracy in the video.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on 1 scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

Also, I use hard negative mining to reduce false positives. I use a 2 second subclip from project video which there're many false positives (shadow of trees etc) and crop the false positive out and use them to retrain my model. Although I only added 43 images to data set, it effectively reduce false positives (at least in the original clip).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Down below shows the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* the model's likely overfitting. The reason being too many features compared with data. Possible improvements inlcuding manually split train and test data to deal with time series issue; and to use more data (e.g. udacity's annoated data)
* improve speed of detection. Currently, it takes about 15 min to process 1261 frames in the project video. Needs to identify which part takes the most time (likely to be sliding window process) and optmize it to be near real time
* explore CNN based detection method (e.g. YOLO, darknet?)
* there are 1 or 2 frames in which the bounding boxes disappear suddenly. Needs to find out why (model issue or smoothing algorithm issue).
