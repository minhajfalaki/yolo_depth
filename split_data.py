import numpy as np
import os
import re
import cv2

def split_data(txt_file):

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
                a=map(int, re.findall(r'\d+', fname))
                print a

                image = cv2.imread(fname,-1)
                image = cv2.resize(image,(480,480))
                print image

                if a[1]<=868:
                    f= open("training_labels1.txt","a+")
                    cv2.imwrite("train1/"+annot[0]+".pgm",image)
                    f.write(each+" \r\n")
                    f.close()

                else:
                    l= open("validation_labels1.txt","a+")
                    cv2.imwrite("validation1/"+annot[0]+".pgm",image)
                    l.write(each+" \r\n")
                    l.close()


            except:
                print "Exception occured at line"
                raise


split_data("labels.txt")