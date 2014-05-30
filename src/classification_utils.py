'''
Classification utils
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import numpy as np
import math_utils as mu
import loader as l
import procrustes_analysis as pa
import fitting_utils as fu
import configuration as c

def get_average_size(method=''):
    IBS = create_individual_bboxes(method)
    Avg = np.zeros((c.get_nb_teeth(), 2))
    for j in range(IBS.shape[0]):
        x = y = 0
        for i in range(IBS.shape[1]):
            x += IBS[j,i,1] - IBS[j,i,0]
            y += IBS[j,i,3] - IBS[j,i,2]
        Avg[j,0] = x / float(IBS.shape[1])
        Avg[j,1] = y / float(IBS.shape[1])
    return Avg      

def get_average_params(trainingSamples, method=''):
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    Params = np.zeros((c.get_nb_teeth(), 4))
    
    for j in range(c.get_nb_teeth()):
        S = XS[j,:,:]
        M, Y = pa.PA(S)
        MS[j,:] = M
        
        mtx = mty = ms = mtheta = 0
        n = S.shape[0]
        for i in range(n):
            tx, ty, s, theta = mu.full_align_params(M, fu.original_to_cropped(S[i,:]))
            mtx += tx
            mty += ty
            ms += s
            mtheta += theta
        n = float(n)
        Params[j,0] = mtx / n
        Params[j,1] = mty / n
        Params[j,2] = ms / n
        Params[j,3] = mtheta / n
    return Params
    
def create_average_models(trainingSamples, method=''):
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    IS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        S = XS[j,:,:]
        M, Y = pa.PA(S)
        MS[j,:] = M
        
        mtx = mty = ms = mtheta = 0
        n = S.shape[0]
        for i in range(n):
            tx, ty, s, theta = mu.full_align_params(M, fu.original_to_cropped(S[i,:]))
            mtx += tx
            mty += ty
            ms += s
            mtheta += theta
        n = float(n)
        IS[j,:] = mu.full_align(M, (mtx / n), (mty / n), (ms / n), (mtheta / n))  

def create_individual_bboxes(method=''):
    XS = l.create_full_XS()
    IBS = np.zeros((XS.shape[0], XS.shape[1], 4))
    for j in range(XS.shape[0]):
        for i in range(XS.shape[1]):
            min_y = min_x = float("inf")
            max_y = max_x = 0
            x_coords, y_coords = mu.extract_coordinates(XS[j,i,:])
            for k in range(x_coords.shape[0]):
                if x_coords[k] < min_x: min_x = x_coords[k]
                if x_coords[k] > max_x: max_x = x_coords[k]
                if y_coords[k] < min_y: min_y = y_coords[k]
                if y_coords[k] > max_y: max_y = y_coords[k]
            IBS[j,i,0] = min_x - fu.offsetX
            IBS[j,i,1] = max_x - fu.offsetX
            IBS[j,i,2] = min_y - fu.offsetY
            IBS[j,i,3] = max_y - fu.offsetY
    return IBS 

def create_bboxes(method=''):
    IBS = create_individual_bboxes(method)
    BS = np.zeros((IBS.shape[1], 8))
    for i in range(IBS.shape[1]):
        min_y = min_x = float("inf")
        max_y = max_x = 0
        for j in range(IBS.shape[0]/2):
            if IBS[j,i,0] < min_x: min_x = IBS[j,i,0]
            if IBS[j,i,1] > max_x: max_x = IBS[j,i,1]
            if IBS[j,i,2] < min_y: min_y = IBS[j,i,2]
            if IBS[j,i,3] > max_y: max_y = IBS[j,i,3]
        BS[i,0] = min_x
        BS[i,1] = max_x
        BS[i,2] = min_y
        BS[i,3] = max_y
   
        min_y = min_x = float("inf")
        max_y = max_x = 0    
        for j in range(IBS.shape[0]/2, IBS.shape[0]):
            if IBS[j,i,0] < min_x: min_x = IBS[j,i,0]
            if IBS[j,i,1] > max_x: max_x = IBS[j,i,1]
            if IBS[j,i,2] < min_y: min_y = IBS[j,i,2]
            if IBS[j,i,3] > max_y: max_y = IBS[j,i,3]
        BS[i,4] = min_x
        BS[i,5] = max_x
        BS[i,6] = min_y
        BS[i,7] = max_y
    return BS

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
        cv2.imwrite(fname, img[max_y-fu.offsetY+1:,:])
        
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
        cv2.imwrite(fname, img[:min_y-fu.offsetY,:])
    
    
def classify_positives(method=''):
    XS = l.create_full_XS()
    
    for s in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(s)
        try:
            info_name_upper = c.get_dir_prefix() + 'data/Visualizations/Classified Samples/info' + method + str(s) + '-u' + '.txt' 
            info_name_lower = c.get_dir_prefix() + 'data/Visualizations/Classified Samples/info' + method + str(s) + '-l' + '.txt' 

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
                
                line = 'rawdata/' + method + img_name + ' 1 ' + str(int(min_x - fu.offsetX)) + ' ' + str(int(min_y - fu.offsetY)) + ' ' + str(int(max_x - min_x)) + ' ' + str(int(max_y - min_y)) + '\n' 
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
                
                line = 'rawdata/' + method + img_name + ' 1 ' + str(int(min_x - fu.offsetX)) + ' ' + str(int(min_y - fu.offsetY)) + ' ' + str(int(max_x - min_x)) + ' ' + str(int(max_y - min_y)) + '\n' 
                info_file_lower.write(line) 

        finally:
            info_file_upper.close()
            info_file_lower.close()

if __name__ == '__main__':
    #create_negatives(method='SCD')
    classify_positives(method='SCD')