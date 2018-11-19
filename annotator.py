from imgaug import augmenters as iaa
from keras.utils import Sequence
# from utils import BoundBox, bbox_iou
from tqdm import tqdm
# import xml.etree.ElementTree as ET
import numpy as np
import imgaug as ia
import os
import copy
import cv2


def parse_annotation_txt(txt_file):

    print("parsing {} txt file can took a while, wait please.".format(txt_file))
    all_imgs = []
    seen_labels = {}

    all_imgs_indices = {}
    count_indice = 0
    with open(txt_file, "r") as annotations:
        read = annotations.read().split('\n')
        for each in read:
            annot = each.split(" ")
            if annot[0] == "": continue
            try:
                fname = 'depth/'+annot[0]+".pgm"

                xmin = int(annot[1])
                ymin = int(annot[2])
                xmax = xmin + int(annot[3])
                ymax = ymin + int(annot[4])
                obj_name="person_in_depth_image"

                               
                img = {'object':[]}
                img['filename'] = fname
                img['width'] = 640
                img['height'] = 480

                if obj_name == "":
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                    continue

                obj = {}
                obj['xmin'] = xmin
                obj['xmax'] = xmax
                obj['ymin'] = ymin
                obj['ymax'] = ymax
                obj['name'] = obj_name

                img['object'].append(obj)


                if fname not in all_imgs_indices:
                    all_imgs_indices[fname] = count_indice
                    all_imgs.append(img)
                    count_indice += 1
                else:
                    all_imgs[all_imgs_indices[fname]]['object'].append(obj)

                if obj_name not in seen_labels:
                    seen_labels[obj_name] = 1
                else:
                    seen_labels[obj_name] += 1

            except:
                print "Exception occured at line {} from {}".format(i+1, txt_file)
                raise

    return all_imgs, seen_labels



a = parse_annotation_txt("labels.txt")
print len(a)
for each in a[0]:
    print each

                

