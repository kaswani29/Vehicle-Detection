from collections import deque
from functools import reduce
import numpy as np


# To keep information of heatmap in previous frames over several frames deque is being used

class HeatTracker(object):

    def __init__(self, image,mysmoothover =5):
        self.sum_heat_img = np.zeros_like(image[:, :, 0]).astype(np.float) #
        self.heatlist = deque([], maxlen=mysmoothover)
        self.smoothover = mysmoothover

    def avg_heat(self, heatimg):

        self.heatlist.append(heatimg)

        # if len(self.heatlist)> 11: # used fixed length deque instead
        #     self.heatlist.popleft()

        return reduce(lambda x,y: x+y, self.heatlist)


# To do: implement a more sophisticated car tracker
# class car_tracker():
#     def __init__(self):
#         self.detected = False # was the vehicle detected in last iterations
#         self.n_detections = 0 # number of times this vehicle has been detected
#         self.n_nondetection = 0 # number of times this vehicle is consecutively not detected
#         self.xpixels = None # Pixel x values of last detection
#         self.ypixels = None # Pixel y values of last detection
#         self.recent_x_fitted = [] # x position of last n fits of bounding box
#         self.recent_y_fitted = [] # y position of last n fits of bounding box
#         self.bestx = None # average x position of last n fits
#         self.besty = None # average y position of last n fits
#



# carslist = []
#
# carslist.append(car_tracker())
#
