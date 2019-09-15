# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:56:38 2019

@author: Meet
"""

import tensorflow as tf
import cv2
import glob

summary_writer = tf.summary.FileWriter("./summary")

files = glob.glob("D:/images/*.jpg")

for c in range(100):
    summary = tf.Summary()
    
    if (c + 1) % 10 == 0: 
        img_file = files[c]
        img = cv2.imread(img_file)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        retval, buffer = cv2.imencode('.jpg', img)
        img_sum = tf.Summary.Image(encoded_image_string=buffer.tostring(),
                                           height=img.shape[0],
                                           width=img.shape[1])
        
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="summary_scalar_value", simple_value=10 * c),
            tf.Summary.Value(tag="summary_image", image=img_sum) 
        ])
    else:
      summary = tf.Summary(value=[
            tf.Summary.Value(tag="summary_scalar_value", simple_value=10 * c),
        ])
    summary_writer.add_summary(summary, c)
    summary_writer.flush()
    
    
