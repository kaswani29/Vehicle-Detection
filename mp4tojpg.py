import cv2

# Test on images from the video

vidcap = cv2.VideoCapture('project_video.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  cv2.imwrite("out/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1