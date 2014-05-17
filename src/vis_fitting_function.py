
import numpy as np
import cv2
import configuration as c


def create_all_gradient_images():
    create_gradient_images('SC')
    create_gradient_images('SCD')
    create_gradient_images('EH')
    create_gradient_images('EHD')

def create_gradient_images(method=''):
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        temp = cv2.Scharr(img, ddepth=-1, dx=1, dy=0)
        gradient = cv2.Scharr(temp, ddepth=-1, dx=0, dy=1)
        fname = c.get_fname_vis_ff_gradients(i, method)
        cv2.imwrite(fname, gradient)

if __name__ == '__main__':
    create_all_gradient_images()