# -*- coding: utf-8 -*-
'''
Some mathematical utilities
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import math
import numpy as np

def normalize_vector(v):
    '''
    Normalize the given vector.
    '''
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm
    
def rotate(v, s=1, theta=0):
    sc = s*math.cos(theta)
    ss = s*math.sin(theta)
    r = np.zeros(v.shape)
    for i in range(v.shape[0] / 2):
        x = v[(2*i)]
        y = v[(2*i+1)]
        r[(2*i)] = sc*x-ss*y
        r[(2*i+1)] = ss*x+sc*y
    return r
    
def align_with(x1, x2):
    s, theta = align_with_params(x1, x2)
    return rotate(x2, s, theta) 
    
def align_with_params(x1, x2):
    n = pow(np.linalg.norm(x1), 2)
    a = np.dot(x1, x2) / n
    b = 0
    for i in range(x1.shape[0] / 2):
        b += x1[(2*i)]*x2[(2*i+1)] - x1[(2*i+1)]*x2[(2*i)]
    b /= n
    
    s = math.sqrt(a*a+b*b)
    theta = math.atan(b/a)
    return (s, theta)