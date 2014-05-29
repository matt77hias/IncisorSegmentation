'''
Classification utils
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import math_utils as mu
import loader as l
import configuration as c

offsetY = 497.0                 #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                #to locate them on the cropped images.
offsetX = 1234.0                #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                #to locate them on the cropped images.
def create_negatives(method=''):
    XS = l.create_full_XS()
    
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        s = ''    
        if (i < 10): s = '0'
        
        min_y = min_x = float("inf")
        max_y = max_x = 0
    
        for j in range(0, c.get_nb_teeth()/2):
            x_coords, y_coords = mu.extract_coordinates(XS[j, i-1, :])
            for k in range(c.get_nb_landmarks()):
                if x_coords[k] < min_x: min_x = x_coords[k]
                if x_coords[k] > max_x: max_x = x_coords[k]
                if y_coords[k] < min_y: min_y = y_coords[k]
                if y_coords[k] > max_y: max_y = y_coords[k]
        
        fname = c.get_dir_prefix() + 'data/Visualizations/Classified Samples/' + method + str(s) + str(i) + '-l' + '.png'    
        cv2.imwrite(fname, img[max_y-offsetY+1:,:])
        
        min_y = min_x = float("inf")
        max_y = max_x = 0
        
        for j in range(c.get_nb_teeth()/2, c.get_nb_teeth()):
            x_coords, y_coords = mu.extract_coordinates(XS[j, i-1, :])
            for k in range(c.get_nb_landmarks()):
                if x_coords[k] < min_x: min_x = x_coords[k]
                if x_coords[k] > max_x: max_x = x_coords[k]
                if y_coords[k] < min_y: min_y = y_coords[k]
                if y_coords[k] > max_y: max_y = y_coords[k] 
        
        fname = c.get_dir_prefix() + 'data/Visualizations/Classified Samples/' + method + str(s) + str(i) + '-u' + '.png'    
        cv2.imwrite(fname, img[:min_y-offsetY,:])
    
    
def classify_positives(method=''):
    XS = l.create_full_XS()
    
    for s in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(s)
        try:
            info_name_upper = c.get_dir_prefix() + 'data/Visualizations/Classified Samples/info ' + method + str(s) + '-u' + '.txt' 
            info_name_lower = c.get_dir_prefix() + 'data/Visualizations/Classified Samples/info ' + method + str(s) + '-l' + '.txt' 

            info_file_upper = open(info_name_upper, "w")
            info_file_lower = open(info_name_lower, "w")
            
            for i in trainingSamples:
                
                s = ''    
                if (i < 10):
                    s = '0'
                img_name = s + str(i) + '.png'
                
                min_y = min_x = float("inf")
                max_y = max_x = 0

                for j in range(0, c.get_nb_teeth()/2):
                    x_coords, y_coords = mu.extract_coordinates(XS[j, i-1, :])
                    for k in range(c.get_nb_landmarks()):
                        if x_coords[k] < min_x: min_x = x_coords[k]
                        if x_coords[k] > max_x: max_x = x_coords[k]
                        if y_coords[k] < min_y: min_y = y_coords[k]
                        if y_coords[k] > max_y: max_y = y_coords[k]
                
                line = 'rawdata/' + method + img_name + ' 1 ' + str(int(min_x - offsetX)) + ' ' + str(int(min_y - offsetY)) + ' ' + str(int(max_x - min_x)) + ' ' + str(int(max_y - min_y)) + '\n' 
                info_file_upper.write(line)
                
                min_y = min_x = float("inf")
                max_y = max_x = 0
                
                for j in range(c.get_nb_teeth()/2, c.get_nb_teeth()):
                    x_coords, y_coords = mu.extract_coordinates(XS[j, i-1, :])
                    for k in range(c.get_nb_landmarks()):
                        if x_coords[k] < min_x: min_x = x_coords[k]
                        if x_coords[k] > max_x: max_x = x_coords[k]
                        if y_coords[k] < min_y: min_y = y_coords[k]
                        if y_coords[k] > max_y: max_y = y_coords[k] 
                
                line = 'rawdata/' + method + img_name + ' 1 ' + str(int(min_x - offsetX)) + ' ' + str(int(min_y - offsetY)) + ' ' + str(int(max_x - min_x)) + ' ' + str(int(max_y - min_y)) + '\n' 
                info_file_lower.write(line) 

        finally:
            info_file_upper.close()
            info_file_lower.close()

if __name__ == '__main__':
    #create_negatives(method='SCD')
    classify_positives(method='SCD')