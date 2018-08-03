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
    @param v:           the vector to make circular
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
    @param v:            the vector to extract the coordinates from
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
    @param v:           the vector to normalize
    @return The normalized vector.
    '''
    norm=np.linalg.norm(v)
    if norm==0: 
        return v
    return v/norm
    
def get_center_of_gravity(v):
    '''
    Returns the center of gravity of the given vector v.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector v
    @return The center of gravity of the given vector v.
    '''
    xs, ys = extract_coordinates(v)
    n = float(v.shape[0] / 2)
    return (sum(xs) / n, sum(ys) / n)
    
def full_align_with(v, t):
    '''
    Fully alligns the given vector v with the given vector t
    without preconditions.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to align
    @param t:           the vector to align with
    @return The fully aligned vector.
    '''
    ov = center_onOrigin(v)
    ot = center_onOrigin(t)
    return center_on(align_with(ov, ot), t)
    
def full_align_params(v, t):
    '''
    Returns the transformation parameters (tx, ty, s, theta) used
    for translating v with tx in the x direction,
    for translating with ty in the u direction,
    for scaling v with s and for rotating v with theta
    in order to align the transformed v with y.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to align
    @param t:           the vector to align with
    @return The transformation parameters (tx, ty, s, theta) for aligning v with t.
    '''
    s, theta = align_params(center_onOrigin(v), center_onOrigin(t))
    txm, tym = get_center_of_gravity(t)
    return txm, tym, s, theta
    
def full_align(v, tx, ty, s, theta):
    '''
    Fully alligns the given vector v with the given vector t
    with the given transformation parameters.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to align.
    @param tx:          the translation parameter in the x direction
    @param ty:          the translation parameter in the y direction
    @param s:           the scaling parameter
    @param theta:       the rotation parameter
    @return The fully aligned vector.
    '''
    return translate(align(center_onOrigin(v), s, theta), tx, ty)
    
def translate(v, tx, ty):
    '''
    Translates the given vector v with tx in x direction
    and ty in y direction.
    @return The translated vector.
    '''
    r = np.zeros(v.shape)
    for i in range(r.shape[0] / 2):
        r[(2*i)] = v[(2*i)] + tx
        r[(2*i+1)] = v[(2*i+1)] + ty
    return r
    
def center_on(v, t):
    '''
    Centers the given vector v on the center of gravity of the given vector t.
    @pre    v is centered on the origin
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to center
    @param t:           the vector to center on
    @return The centered vector.
    '''
    txm, tym = get_center_of_gravity(t)
    return translate(v, txm, tym)
    
def center_onOrigin(v):
    '''
    Centers the given vector on the origin.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param v:           the vector to center on the origin
    @return The centered vector.
    '''
    xm, ym = get_center_of_gravity(v)
    return translate(v, -xm, -ym)
    
def align_with(v, t):
    '''
    Aligns v with t
    @pre    v and t are centered on the origin
    @pre    The landmark coordinates are stored as successive xi, yi, xj, yj, ...
    @param v           the vector to align
    @param t           the vector to align with
    @return The aligned vector.
    '''
    s, theta = align_params(v, t)
    return align(v, s, theta) 
    
def align_params(v, t):
    '''
    Returns the transformation parameters (s, theta) used for scaling
    v with s and for rotating v with theta in order to align the transformed
    v with t. This method tries to minimize |s.R(theta).v-t|.
    @pre    v and t are centered on the origin
    @pre    The landmark coordinates are stored as successive xi, yi, xj, yj, ...
    @param v           the vector to align
    @param t           the vector to align with
    @return The transformation parameters (s, theta) for aligning v with t.
    '''
    n = pow(np.linalg.norm(v), 2)
    a = np.dot(v, t) / n
    b = 0
    for i in range(v.shape[0] / 2):
        # landmark coordinates stored as successive xi, yi, xj, yj
        b += v[(2*i)]*t[(2*i+1)] - v[(2*i+1)]*t[(2*i)]
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
    @return The aligned vector.
    '''
    sc = s*math.cos(theta)
    ss = s*math.sin(theta)
    r = np.zeros(v.shape)
    for i in range(v.shape[0] / 2):
        # landmark coordinates stored as successive xi, yi, xj, yj
        x = v[(2*i)]
        y = v[(2*i+1)]
        r[(2*i)] = sc*x-ss*y
        r[(2*i+1)] = ss*x+sc*y
    return r
