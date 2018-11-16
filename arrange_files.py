from os import listdir
import os
from os.path import isfile, join
import re
import numpy as np
from PIL import Image
import cv2


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# f=[os.path.abspath(x) for x in os.listdir("/home/minhajf/mensa_seq0_1.1/depth")]
# print f

i=0
for f in listdir("/home/minhajf/RGBD/rgb"):
	# f = os.path.abspath(f)
	a=map(int, re.findall(r'\d+', f))
	print i,f,a
	# if a[2]
	im=cv2.imread("/home/minhajf/RGBD/rgb/"+f)
	print im
	
	# if im is not None:
	print im.shape,"ppm"
	img=rotateImage(im,90)
	# print img.shape,"a"
	cv2.imwrite("/home/minhajf/RGBD/rgb_r/%s/img_%s.png"%(a[2],a[1]), img)

	i+=1
