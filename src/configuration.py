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
dir_mirrored = "data/Landmarks/mirrored"
dir_original = "data/Landmarks/original"

nb_trainingSamples = 14     #from 1 to 14
nb_teeth = 8                #from 1 to 8
nb_landmarks = 40           #from 1 to 40 (points representing an example)
nb_dim = nb_landmarks*2

def get_dir_prefix():
    if (os.environ.get("USERNAME") == "Milan Samyn"):
        return "../"
    else:
        return "CV/"

def get_dir_radiographs():
    return get_dir_prefix() + dir_radiographs
    
def get_dir_mirrored_landmarks():
    return get_dir_prefix() + dir_mirrored

def get_dir_original_landmarks():
    return get_dir_prefix() + dir_original
    
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