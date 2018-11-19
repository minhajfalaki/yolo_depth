from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
import copy
import sys
import shutil
import cv2


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        # print xmin,xmax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1
        # print "inside BoundBox"

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        # print self.label
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        # print self.score
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect

    # print intersect, union    
    return float(intersect) / union

def draw_boxes(image, boxes, labels):
    image_h, image_w = image.shape

    color_levels = [0,255,128,64,32]
    colors = []
    for r in color_levels:
        for g in color_levels:
            for b in color_levels:
                if r==g and r==b: #prevent grayscale colors
                    continue
                colors.append((b,g,r))

    for box in boxes:
        # print box
        xmin = (box.xmin*image_w)*100
        ymin = (box.ymin*image_h)*100
        xmax = (box.xmax*image_w)*100
        ymax = (box.ymax*image_h)*100

        line_width_factor = (min(image_h,image_w)*0.005)
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), colors[box.get_label()], line_width_factor*2)
        cv2.putText(image, 
                    "{} {:.3f}".format(labels[box.get_label()],box.get_score()),  
                    (xmin, ymin - line_width_factor * 3), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    2e-3 * min(image_h,image_w), 
                    (0,255,0), line_width_factor)
        
    return image      
        
def decode_netout(netout, anchors, nb_class, obj_threshold=0.02, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    # netout[..., 5:] *= netout[..., 5:] > obj_threshold
    print netout[..., 5:]
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                classes = netout[row,col,b,5:]

                
                if np.sum(classes) == 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x))*10 / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y))*10 / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w)*10 / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h)*10 / grid_h # unit: image height

                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    
                    boxes.append(box)


    for c in range(nb_class):

        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))


        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            
            if boxes[index_i].classes[c] != 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    # print bbox_iou(boxes[index_i], boxes[index_j]),"nms"
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_i].classes[c] = 0
                        

    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    print boxes
    
    return boxes    

   

        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

