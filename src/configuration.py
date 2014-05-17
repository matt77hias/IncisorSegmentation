# -*- coding: utf-8 -*-
'''
Configuration
Contains all global parameters and directories
used and accessed in the other .py files
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''
import os;

dir_radiographs = "data/Radiographs"
dir_training_radiographs = "data/Radiographs/Training"
dir_test_radiographs = "data/Radiographs/Test"
dir_mirrored = "data/Landmarks/mirrored"
dir_original = "data/Landmarks/original"
dir_fitting_manual = "data/Fitting/Manual"
dir_pyramid = "data/Radiographs/Test/Pyramid"

#Own visualizations
dir_vis_landmarks = "data/Visualizations/Landmarks"
dir_vis_pa = "data/Visualizations/PA"
dir_vis_pca = "data/Visualizations/PCA"
dir_vis_pre = "data/Visualizations/Preproccess"
dir_vis_ff_gradients = "data/Visualizations/Fitting Function/Gradients"
dir_vis_ff_landmarks = "data/Visualizations/Fitting Function/Landmarks"
dir_vis_ff_landmarks_and_models = "data/Visualizations/Fitting Function/Landmarks and Models"
dir_vis_ff_models = "data/Visualizations/Fitting Function/Models"
dir_vis_ff_profile_normals = "data/Visualizations/Fitting Function/Profile Normals"

nb_trainingSamples = 14     #from 1 to 14
nb_testSamples = 16         #from 15 to 30

#TODO: nagaan hoeveel training samples we in feite nodig hebben (door te testen hoe goed het model generaliseerd (zie pag 10 ASM)

nb_teeth = 8                #from 1 to 8
nb_landmarks = 40           #from 1 to 40 (points representing an example)
nb_dim = nb_landmarks*2

#Directories

def get_dir_prefix():
    if (os.environ.get("USERNAME") == "Milan Samyn"):
        return "../"
    else:
        return "CV/"

def get_dir_radiographs():
    return get_dir_prefix() + dir_radiographs

def get_dir_test_radiographs():
    return get_dir_prefix() + dir_test_radiographs
    
def get_dir_mirrored_landmarks():
    return get_dir_prefix() + dir_mirrored

def get_dir_original_landmarks():
    return get_dir_prefix() + dir_original
    
def get_dir_fitting_manual():
    return get_dir_prefix() + dir_fitting_manual

def get_dir_pyramid():
    return get_dir_prefix() + dir_pyramid
    
def get_dir_vis_landmarks():
    return get_dir_prefix() + dir_vis_landmarks
    
def get_dir_vis_pa():
    return get_dir_prefix() + dir_vis_pa
    
def get_dir_vis_pca():
    return get_dir_prefix() + dir_vis_pca 

def get_dir_vis_pre():
    return get_dir_prefix() + dir_vis_pre    
    
def get_dir_vis_ff_gradients():
    return get_dir_prefix() + dir_vis_ff_gradients  
    
def get_dir_vis_ff_landmarks():
    return get_dir_prefix() + dir_vis_ff_landmarks  
    
def get_dir_vis_ff_landmarks_and_models():
    return get_dir_prefix() + dir_vis_ff_landmarks_and_models 
    
def get_dir_vis_ff_models():
    return get_dir_prefix() + dir_vis_ff_models  
    
def get_dir_vis_ff_profile_normals():
    return get_dir_prefix() + dir_vis_ff_profile_normals 

#File names
  
def get_fname_radiograph(nr_trainingSample):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = "/"
    if (nr_trainingSample < 10):
        s = "/0"
    fname = (get_dir_radiographs() + s + str(nr_trainingSample) + '.tif')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname

'''
TODO: opsplitsen voor test en training samples best
'''

def get_fname_test_radiograph(nr_testSample):
    
    new_nr_testSample = nr_testSample + get_nb_testSamples() - 1
    s = "/"
    if (new_nr_testSample < 10):
        s = "/0"
    fname = (get_dir_test_radiographs() + s + str(new_nr_testSample) + '.tif')
    
    if (not is_valid_testSample(nr_testSample)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_mirrored_landmark(nr_trainingSample, nr_tooth):
    fname = (get_dir_mirrored_landmarks() + "/landmarks" + str(nr_trainingSample) + '-' + str(nr_tooth) + '.txt')
    
    if (not is_valid_trainingSample(nr_trainingSample) or not is_valid_tooth(nr_tooth)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_original_landmark(nr_trainingSample, nr_tooth):
    fname = (get_dir_original_landmarks() + "/landmarks" + str(nr_trainingSample) + '-' + str(nr_tooth) + '.txt')
    
    if (not is_valid_trainingSample(nr_trainingSample) or not is_valid_tooth(nr_tooth)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth):
    fname = (get_dir_fitting_manual() + "/landmarks" + str(nr_trainingSample) + '-' + str(nr_tooth) + '.txt')
    
    if (not is_valid_trainingSample(nr_trainingSample) or not is_valid_tooth(nr_tooth)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_pyramid(nr_testSample, level):
    new_nr_testSample = nr_testSample + get_nb_testSamples() - 1
    fname = (get_dir_pyramid() + "/pyramid" + str(new_nr_testSample) + '-' + str(level) + '.png')
    
    if (not is_valid_testSample(nr_testSample)):
        raise InvalidFileName(fname)
    
    return fname

def get_fname_vis_landmark(nr_trainingSample, nr_tooth):
    fname = (get_dir_vis_landmarks() + "/landmarks" + str(nr_trainingSample) + '-' + str(nr_tooth) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample) or not is_valid_tooth(nr_tooth)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_vis_pa(nr_tooth, samples_included=False):
    if samples_included:
        fname = (get_dir_vis_pa() + "/mean" + str(nr_tooth) + '-s.png')
    else:
        fname = (get_dir_vis_pa() + "/mean" + str(nr_tooth) + '.png')
    
    if (not is_valid_tooth(nr_tooth)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_vis_pca(nr_tooth, nr_eig):
    fname = (get_dir_vis_pca() + "/eig" + str(nr_tooth) + '-' + str(nr_eig) + '.png')
    
    if (not is_valid_tooth(nr_tooth)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_vis_pre(nr_trainingSample, method=''):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = '/' + method
    if (nr_trainingSample < 10):
        s = '/' + method + '0'
    fname = (get_dir_vis_pre() + s + str(nr_trainingSample) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_ff_gradients(nr_trainingSample, method=''):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = '/' + method
    if (nr_trainingSample < 10):
        s = '/' + method + '0'
    fname = (get_dir_vis_ff_gradients() + s + str(nr_trainingSample) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_ff_landmarks(nr_trainingSample, method=''):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = '/' + method
    if (nr_trainingSample < 10):
        s = '/' + method + '0'
    fname = (get_dir_vis_ff_landmarks() + s + str(nr_trainingSample) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_ff_landmarks_and_models(nr_trainingSample, method=''):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = '/' + method
    if (nr_trainingSample < 10):
        s = '/' + method + '0'
    fname = (get_dir_vis_ff_landmarks_and_models() + s + str(nr_trainingSample) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_ff_models(nr_trainingSample, method=''):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = '/' + method
    if (nr_trainingSample < 10):
        s = '/' + method + '0'
    fname = (get_dir_vis_ff_models() + s + str(nr_trainingSample) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname
    
def get_fname_ff_profile_normals(nr_trainingSample, method=''):
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName
    
    s = '/' + method
    if (nr_trainingSample < 10):
        s = '/' + method + '0'
    fname = (get_dir_vis_ff_profile_normals() + s + str(nr_trainingSample) + '.png')
    
    if (not is_valid_trainingSample(nr_trainingSample)):
        raise InvalidFileName(fname)
    
    return fname
    
#Numbers and ranges

def get_nb_trainingSamples():
    return nb_trainingSamples
    
def get_trainingSamples_range():
    return range(1, (get_nb_trainingSamples()+1))  

def get_nb_testSamples():
    return nb_testSamples
    
def get_testSamples_range():
    return range(1, (get_nb_testSamples()))   
    
def get_nb_teeth():
    return nb_teeth
    
def get_teeth_range():
    return range(1, (get_nb_teeth()+1))
    
def get_nb_landmarks():
    return nb_landmarks
    
def get_nb_dim():
    return nb_dim
 
#Validation

def is_valid_trainingSample(nr_trainingSample):
    return (nr_trainingSample in get_trainingSamples_range())

def is_valid_testSample(nr_testSample):
    return (nr_testSample in get_testSamples_range())
    
def is_valid_tooth(nr_tooth):
    return (nr_tooth in get_teeth_range())
    
class InvalidFileName(Exception):
     def __init__(self, value):
        self.value = value
     def __str__(self):
        return repr(self.value)