# -*- coding: utf-8 -*-
'''
Some mathematical utilities
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import math
import numpy as np

def make_circular(v):
    '''
    Makes the given vector circular by appending the first entry.
    @param v:           the vector to make circular.
    @return The circular vector.
    '''
    w = np.zeros((v.shape[0]+1))
    w[:-1] = v
    w[-1] = v[0]
    return w
    
def extract_coordinates(v):
    '''
    Extracts the x and y coordinates from the given vector.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:            the vector to extract the coordinates from.
    @return The x and y coordinates extracted from the given vector.
    '''
    n = v.shape[0] / 2
    xCoords = np.zeros(n)
    yCoords = np.zeros(n)
    for i in range(n):
        xCoords[i] = v[(2*i)]  
        yCoords[i] = v[(2*i+1)]
    return xCoords, yCoords
    
def zip_coordinates(xCoords, yCoords):
    '''
    Zips the given x and y coordinates.
    @param xCoords:     the x coordinates
    @param yCoords:     the y coordinates
    @return The zipped vector.
    '''
    n = xCoords.shape[0]
    v = np.zeros(2 * n)
    for i in range(n):
        v[(2*i)] = xCoords[i]  
        v[(2*i+1)] = yCoords[i]
    return v

def normalize_vector(v):
    '''
    Normalize the given vector.
    @param v:           the vector to normalize.
    @return The normalized vector.
    '''
    norm=np.linalg.norm(v)
    if norm==0: 
        return v
    return v/norm
    
def center_onOrigin(v):
    '''
    Centers the given vector on the origin.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to center on the origin.
    @return The centered vector.
    '''
    xCoords, yCoords = extract_coordinates(v)
    n = v.shape[0] / 2
    xm = sum(xCoords) / n
    ym = sum(yCoords) / n
    for i in range(n):
        xCoords[i] -= xm
        yCoords[i] -= ym
    return zip_coordinates(xCoords, yCoords)
    
def align_with(x1, x2):
    '''
    Aligns x1 with x2
    @pre    x1 and x2 are centered on the origin
    @pre    The landmark coordinates are stored as successive xi, yi, xj, yj, ...
    @param x1           the vector to align
    @param x2           the vector to align with
    @return The aligned vector for x1
    '''
    s, theta = align_params(x1, x2)
    return align(x1, s, theta) 
    
def align_params(x1, x2):
    '''
    Returns the transformation parameters (s, theta) used for scaling
    x1 with s and rotating x1 with theta in order to align the transformed
    x1 with x2. This method tries to minimize |s.R(theta).x1-x2|.
    @pre    x1 and x2 are centered on the origin
    @pre    The landmark coordinates are stored as successive xi, yi, xj, yj, ...
    @param x1           the vector to align
    @param x2           the vector to align with
    @return The transformation parameters (s, theta) for aligning x1 with x2.
    '''
    n = pow(np.linalg.norm(x1), 2)
    a = np.dot(x1, x2) / n
    b = 0
    for i in range(x1.shape[0] / 2):
        #landmark coordinates stored as successive xi, yi, xj, yj
        b += x1[(2*i)]*x2[(2*i+1)] - x1[(2*i+1)]*x2[(2*i)]
    b /= n
    
    s = math.sqrt(a*a+b*b)
    theta = math.atan(b/a)
    return (s, theta)
    
def align(v, s=1, theta=0):
    '''
    Aligns x1 with scaling parameter s and rotation parameter theta.
    @pre    v is centered on the origin
    @pre    The landmark coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to align
    @param s:           the scaling parameter
    @param theta:       the rotation parameter
    @return The aligned vector for v
    '''
    sc = s*math.cos(theta)
    ss = s*math.sin(theta)
    r = np.zeros(v.shape)
    for i in range(v.shape[0] / 2):
        #landmark coordinates stored as successive xi, yi, xj, yj
        x = v[(2*i)]
        y = v[(2*i+1)]
        r[(2*i)] = sc*x-ss*y
        r[(2*i+1)] = ss*x+sc*y
    return r