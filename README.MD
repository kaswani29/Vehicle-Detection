**Vehicle Detection Project**

Overview:

* Performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
=======

## **Vehicle Detection Project**

###Overview:

* Extracted Histogram of Oriented Gradients (HOG) features, color transforms features,  binned color features, as well as histograms of color extraction on a labeled training set of images
* Trained classifier such as Linear SVM and Random forest on the concatenated above features for each image
* Implemented a sliding-window technique and used the trained classifier to search for vehicles in images
* Ran the pipeline on a video stream (project_video.mp4) to create a heat map of recurring detections frame by frame and to reject outliers and to follow the detected vehicles.
* Estimated a bounding box for vehicles detected.
[//]: # (Image References)
[image1]: ./examples/trainingdata.png "Training data"
[image2]: ./examples/hogimage.png "HOG features"
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4



###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features:

To extract HOG features and various other features I used the function extrac_features. It is defined in the `feature_functions.py` from 52 to 101. I am using this function in the main project script `vehicle_detection_pipeline.py` in line 29 to 40 to extract features from the training data. I did most of the visualization of different features in `visulaizations.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Look into `visualizations.py`, in lines 44 to 58 where I am using single_img_features function to extract features from a single image at a time.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Optimizing Parameters for HOG, color and spatial binning

I tried various combinations of parameters, explored visualizing them and finally tested using the classifier. The combination that gave best accuracy on test set was chosen in the end. More thorough way to optimize these would be by doing a random grid search using scikit learn.

####3. Training Classifier:

I trained both a linear SVM and Random forest classifer. Both of them had very comparable performance on the test set with SVM doing slightly better. These are implemented from line 60 to 79 in  `vehicle_detection_pipeline.py`

###Sliding Window Search

####1. Implementation of sliding window search, choosing the size and overlap:

For sliding window, I chose to decide only an area from 400 px to 656 px on y axis as above (sky) and below no cars would be found. I kept the entire x axis for the search. For the size of the window, I chose three sizes (64,64), (96,96) and (128,128). 

![alt text][image3]

####2. Pipeline:

Ultimately I chose YCrCb color space with 3-channel HOG features plus spatially binned (32,32) color and histograms of color (32 bins) in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Vehicle detection on Video Pipeline:

Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The search can be further by keeping different y axis for different window sizes.

