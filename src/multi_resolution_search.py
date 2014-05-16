'''
Multi-Resolution Search Algorithm
Improve the efficiency and robustness of the ASM algorithm 
by implementing it in a multi-resolution framework.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

lmax = 3 #Coarsest level of gaussian pyramid (depends on the size of the object in the image)
ns = 2 #Number of sample points either side of current point
nmax = 5 #Maximum number of iterations allowed at each level
pclose = 0.9 #Desired proportion of points found within ns/2 of current position


