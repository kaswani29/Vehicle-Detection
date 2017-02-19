import glob
from feature_functions import *

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:

cars = glob.glob('dataset/vehicles/*/**.png', recursive=True)
notcars = glob.glob('dataset/non-vehicles/*/**.png', recursive=True)

##### Feature space visulaization############
# Plots

# Random image

car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Reading in car image
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plotting trainging dataset
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(car_image)
ax1.set_title('Car image', fontsize=30)
ax2.imshow(notcar_image)
ax2.set_title('Not car image', fontsize=30)

f.savefig('examples/trainingdata.png')

# Feature Parameters
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
                                                  hist_bins=hist_bins, orient=orient,
                                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                  hog_channel=hog_channel,
                                                  spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,
                                                  vis=True)

notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space,
                                                        spatial_size=spatial_size,
                                                        hist_bins=hist_bins, orient=orient,
                                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                                        hog_channel=hog_channel,
                                                        spatial_feat=spatial_feat, hist_feat=hist_feat,
                                                        hog_feat=hog_feat, vis=True)

images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
titles = ["car image", "car HOG image", "notcar image", "notcar HOG image"]

fig = plt.figure(figsize=(12, 3))

visualize(fig, 1, 4, images, titles)

# Alternative way
# # Plotting trainging dataset
# f, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(12,3))
# ax1.imshow(car_image)
# ax1.set_title('car image', fontsize=18)
# ax2.imshow(car_hog_image, cmap= "hot")
# ax2.set_title('car Hog image', fontsize=18)
# ax3.imshow(notcar_image)
# ax3.set_title('not car image', fontsize=18)
# ax4.imshow(notcar_hog_image, cmap= "hot")
# ax4.set_title('not car Hog image', fontsize=18)
#
#
# f.savefig('examples/hogimage.png')
