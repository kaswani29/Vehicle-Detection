import glob
from time import time

from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from time import time
from feature_functions import *

star_time = time()

# Read in cars and notcars
cars = glob.glob('dataset/vehicles/*/**.png', recursive=True)
notcars = glob.glob('dataset/non-vehicles/*/**.png', recursive=True)

# Feature Parameters
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time()
svc.fit(X_train, y_train)
t2 = time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
#


# # Use a random forest
# t = time()
# rf = RandomForestClassifier(n_estimators=50)
# rf.fit(X_train,y_train)
# t2 = time()
# print(round(t2 - t, 2), 'Seconds to train RF...')
# # Check the score of the SVC
# print('Test Accuracy of RF = ', round(rf.score(X_test, y_test), 4))
#
#
# Box using sliding window

# Testing on images
#Uncomment to test on images

test_images = glob.glob('test_images/*.jpg')
images = []
titles = []
y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
over_lap =0.5
#
# for image in test_images:
#
#     t = time()
#     img = mpimg.imread(image)
#     draw_image = np.copy(img)
#     img = img.astype(np.float32)/255 # image trained is .png 0 to 1, image searched is 0 to 255
#     heat = np.zeros_like(img[:, :, 0]).astype(np.float)
#
#     window1 = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
#                            xy_window=(64, 64), xy_overlap=(over_lap, over_lap))
#
#     window2 = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
#                            xy_window=(96, 96), xy_overlap=(over_lap, over_lap))
#     window3 = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
#                            xy_window=(128, 128), xy_overlap=(over_lap, over_lap))
#
#     windows = window1 + window3 + window2
#
#     hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
#                                  spatial_size=spatial_size, hist_bins=hist_bins,
#                                  orient=orient, pix_per_cell=pix_per_cell,
#                                  cell_per_block=cell_per_block,
#                                  hog_channel=hog_channel, spatial_feat=spatial_feat,
#                                  hist_feat=hist_feat, hog_feat=hog_feat)
#     heat_img  = add_heat(heat,hot_windows)
#     heat_img1 = apply_threshold(heat_img,0)
#     heatmap = np.clip(heat_img1, 0, 255)
#     labels = label(heatmap)
#     window_img = draw_labeled_bboxes(np.copy(draw_image), labels)
#     window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
#     images.append(window_img)
#     images.append(heatmap)
#     titles.append(" ")
#     titles.append(" ")
#     print(time()-t, " seconds to process one image with ", len(windows), " windows")

fig = plt.figure(figsize=(12,18))

visualize(fig, 3,2,images, titles)


### funciton generation for video ####
from car_tracker import HeatTracker

car_ind = np.random.randint(0, len(cars))
car_image = mpimg.imread(cars[car_ind])

alpha = HeatTracker(image=car_image, mysmoothover=25)


def find_vehicles(img):
    # t = time()
    # img = mpimg.imread(image)
    draw_image = np.copy(img)
    img = img.astype(np.float32) / 255  # image trained is .png 0 to 1, image searched is 0 to 255
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    window1 = slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 656],
                           xy_window=(64, 64), xy_overlap=(0.5, 0.5))

    window2 = slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 656],
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    window3 = slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 656],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    windows = window1 + window2 + window3

    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    heat_img = add_heat(heat, hot_windows)
    heat_image_sum = alpha.avg_heat(heat_img)
    heat_img1 = apply_threshold(heat_image_sum, 8)
    heatmap = np.clip(heat_img1, 0, 255)
    labels = label(heatmap)
    window_img = draw_labeled_bboxes(draw_image, labels)
    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    return window_img


from moviepy.editor import VideoFileClip

test_out = "heat_avg_smooth25_thesh8.mp4"

clip = VideoFileClip("project_video.mp4")

test_clip = clip.fl_image(find_vehicles)

test_clip.write_videofile(test_out, audio=False)
