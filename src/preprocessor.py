'''
Preprocess the dental radiographs by cropping to a roughly estimated region of interest, 
reducing noise and enhancing contrast
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import cv2
import configuration as config

def crop(image, width=1500, height=1500, top_offset=0):
    '''
    todo
    '''

    [curr_height, curr_width] = image.shape[:2]

    left   = np.around((curr_width - width) / 2)
    right  = left + width
    top    = np.around((curr_height - height) / 2) + top_offset
    bottom = top + height - top_offset
    
    return image[top:bottom, left:right]

if __name__ == '__main__':
    
    image_path = config.get_dir_prefix() + "data/Radiographs/07.tif"
    image = cv2.imread(image_path)
    cropped_image = crop(image, width=1000, height=1100, top_offset=400)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)