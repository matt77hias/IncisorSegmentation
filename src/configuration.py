# -*- coding: utf-8 -*-
'''
Configuration
Contains all global parameters and directories
used and accessed in the other .py files
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

dir_radiographs = "CV/data/Radiographs"
dir_mirrored = "CV/data/Landmarks/mirrored"
dir_original = "CV/data/Landmarks/original"

nb_trainingSamples = 14     #from 1 to 40
nb_teeth = 8                #from 1 to 8
nb_landmarks = 40
nb_dim = nb_landmarks*2

def get_dir_radiographs():
    return dir_radiographs
    
def get_dir_mirrored_landmarks():
    return dir_mirrored

def get_dir_original_landmarks():
    return dir_original
    
def get_nb_trainingSamples():
    return nb_trainingSamples
    
def get_trainingSamples_range():
    return range(1, (get_nb_trainingSamples()+1))
    
def get_nb_teeth():
    return nb_teeth
    
def get_teeth_range():
    return range(1, (get_nb_teeth()+1))
    
def get_nb_landmarks():
    return nb_landmarks
    
def get_nb_dim():
    return nb_dim