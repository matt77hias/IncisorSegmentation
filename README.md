# Incisor Segmentation

Course Computer Vision: a model-based procedure capable of segmenting the incisors in panoramic dental radiographs using an Active Shape Model (ASM). 

**Team**:
* [Matthias Moulin](https://github.com/matt77hias) (Computer Science)
* [Milan Samyn](https://github.com/MilanSamyn) (Computer Science)

**Academic Year**: 2014-2015 (2nd semester - 2nd Master of Science in Engineering: Computer Science)

## About
The purpose of the final project for the course Computer Vision is the development of a model based procedure for segmenting the upper and lower incisor tooths in dental radiographs. To achieve this, we construct an Active Shape Model (ASM) for each of the eight incisors.

### Input Landmarks
#### Image space
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks1.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks2.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks3.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks4.png" width="215"></p>

### Procrustes Analysis (PA)
Before generating an Active Shape Model, each tooth shape (described by its landmarks) of the set of training samples belonging to same tooth needs to be aligned in the same coordinate system. First, we remove the translation component by centering each tooth shape's center of gravity at the origin. Second, we use a Procrustes Analys to align (scalar and rotation component) each tooth shape of the set training samples belonging to same tooth in such a way that the sum of the distances between the aligned tooth shape and the mean aligned tooth shape is minimized. [Cootes92], [Cootes00]

This is an iterative process. In each iteration step the mean aligned tooth shape is recalculated until the difference between two consecutive mean aligned tooth shapes in each dimension is smaller than a threshold value of 10^-5. Initially, we used a larger value but since convergence was reached after one or two iteration steps, we prefer a smaller value (more precision). With a threshold value of 10^-5, convergence is reached after three iteration steps on average.

The resulting tooth models and tooth model landmarks are visualized below for each of the eight incisors together with the landmarks of 14 training samples in the same model (coordinate) space (x', y'). Note that the reference shape for aligning can be choosen arbitrarily with regard to the scaling and rotation component. We have choosen to align with each of the eight tooth shapes of training sample 1.

<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean1-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean2-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean3-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean4-s.png" width="215"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean5-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean6-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean7-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean8-s.png" width="215"></p>

### Principal Component Analysis (PCA)
After the Procrustes Analysis, we have one model tooth for each of the eight incisor tooths in model space *(x', y')*. Each of these models is representative for the average shape of the corresponding incisor tooths. The aligned tooth shapes constitute a distribution in the *2n* dimensional space (with *n* the number of *2*-dimensional landmarks). We want to model this distribution.

We can for example use a basis for the *2n* dimensional space to represent each vector in this space as a *2n* dimensional coefficient vector (*2n* formfactors). By applying a Principal Component Analysis (PCA), we can try to reduce the dimensionality by computing the principal axes (of which the magnitude is determined by the largest eigenvalues of the covariance matrix) of the distribution in order to express vectors approximately in this *2n* dimensional space with less formfactors.

Based on the results for the Principal Component Analysis, it suffice to use for each of the eight incisors only 5, 6 or 7 formfactors (versus *2n* = 40) to describe at least 98% of the variance of the landmark positions in the training set with regard to the corresponding mean model tooth. Alternatively, one can choose the formfactors in such a way that each training sample can be represnted with a certain accuracy.

By varying the formfactors, we can vary the shape of the tooth. We need to determine upper en lower bounds for these variations in order to obtain plausible tooths. We allow +-3 standard deviations (*+-3 sqrt(lambda_i)*) from the mean model tooth. the effect of *-3 sqrt(lambda_i), -2 sqrt(lambda_i), -1 sqrt(lambda_i), 0, +1 sqrt(lambda_i), +2 sqrt(lambda_i), +3 sqrt(lambda_i)* deviations on each formfactor as opposed to the mean tooth model for each of the eight incisors in model space *(x', y')* is visualized below.

#### (reduced) model space tooth 1
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig1-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig1-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig1-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig1-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig1-5.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig1-6.png" width="286"></p>

#### (reduced) model space tooth 2
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig2-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig2-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig2-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig2-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig2-5.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig2-6.png" width="286"></p>

#### (reduced) model space tooth 3
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig3-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig3-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig3-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig3-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig3-5.png" width="286"></p>

#### (reduced) model space tooth 4
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig4-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig4-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig4-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig4-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig4-5.png" width="286"></p>

#### (reduced) model space tooth 5
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-5.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-6.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig5-7.png" width="286"></p>

#### (reduced) model space tooth 6
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-5.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-6.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig6-7.png" width="286"></p>

#### (reduced) model space tooth 7
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-5.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-6.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig7-7.png" width="286"></p>

#### (reduced) model space tooth 8
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-1.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-2.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-3.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-4.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-5.png" width="286"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-6.png" width="286"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PCA/eig8-7.png" width="286"></p>

### Pre-processing
<p align="center"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/O01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/D01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/EH01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/EHD01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/SC01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/SCD01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/S01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/SD01.png" width="107"></p>

1. Original Greyscale
2. Denoising
3. Histogram Equalization
4. Denoising -> Histogram Equalization
5. Linear Contrast Stretching
6. Denoising -> Linear Contrast Stretching
7. Sigmoid
8. Denoising -> Sigmoid

### Fitting functions

### Multi-Resolution
