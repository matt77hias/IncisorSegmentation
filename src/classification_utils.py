'''
Classification utils
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import preprocessor as pre
import math_utils as mu
import numpy as np
import loader as l
import configuration as c

offsetY = 497.0                 #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                #to locate them on the cropped images.
offsetX = 1234.0                #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                #to locate them on the cropped images.

def classify_positives(method='', offset_x=0, offset_y=0):
    XS = l.create_full_XS()
    
    for s in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(s)
        try:
            info_name = '../data/Visualizations/Classified Samples/info ' + method + str(s) + '-u' + '.txt' 
            info_file = open(info_name, "w")
            
            for i in trainingSamples:
              
                s = ''    
                if (i < 10):
                    s = '0'
                img_name = s + str(i) + '.png'
                
                min_x = float("inf")
                max_x = 0
                min_y = float("inf")
                max_y = 0
                    
                for j in range(0, c.get_nb_teeth()/2):
                    x_coords, y_coords = mu.extract_coordinates(XS[j, i-1, :])
                    for k in range(c.get_nb_landmarks()):
                        if x_coords[k] < min_x: min_x = x_coords[k]
                        if x_coords[k] > max_x: max_x = x_coords[k]
                        if y_coords[k] < min_y: min_y = y_coords[k]
                        if y_coords[k] > max_y: max_y = y_coords[k] 
                try:
                    line = 'rawdata/' + method + img_name + ' 1 ' + str(min_x - offset_x) + ' ' + str(min_y - offset_y) + ' ' + str(max_x - min_y) + ' ' + str(max_y - min_y) + '\n' 
                    info_file.write(line) 
                except (IOError):
                    pass
                
        finally:
            info_file.close()
    
        try:
                info_name = '../data/Visualizations/Classified Samples/info ' + method + str(s) + '-l' + '.txt' 
                info_file = open(info_name, "w")
                
                for i in trainingSamples:
                  
                    s = ''    
                    if (i < 10):
                        s = '0'
                    img_name = s + str(i) + '.png'
                    
                    min_x = float("inf")
                    max_x = 0
                    min_y = float("inf")
                    max_y = 0
                        
                    for j in range(c.get_nb_teeth()/2, c.get_nb_teeth()):
                        x_coords, y_coords = mu.extract_coordinates(XS[j, i-1, :])
                        for k in range(c.get_nb_landmarks()):
                            if x_coords[k] < min_x: min_x = x_coords[k]
                            if x_coords[k] > max_x: max_x = x_coords[k]
                            if y_coords[k] < min_y: min_y = y_coords[k]
                            if y_coords[k] > max_y: max_y = y_coords[k] 
                    try:
                        line = 'rawdata/' + method + img_name + ' 1 ' + str(min_x - offset_x) + ' ' + str(min_y - offset_y) + ' ' + str(max_x - min_y) + ' ' + str(max_y - min_y) + '\n' 
                        info_file.write(line) 
                    except (IOError):
                        pass
                    
        finally:
            info_file.close()
                 
        
    
    


if __name__ == '__main__':

    classify_positives(method='SCD', offset_x=1234.0, offset_y=497.0)