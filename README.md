# Incisor Segmentation

Course Computer Vision: a model-based procedure capable of segmenting the incisors in panoramic dental radiographs using an Active Shape Model (ASM) at multiple resolutions. 

**Team**:
* [Matthias Moulin](https://github.com/matt77hias) (Computer Science)
* [Milan Samyn](https://github.com/MilanSamyn) (Computer Science)

**Academic Year**: 2013-2014 (2nd semester - 1st Master of Science in Engineering: Computer Science)

## About
The goal of the final project for the course Computer Vision is the development of a model-based procedure capable of segmenting the upper and lower incisor teeth in panoramic dental radiographs. To achieve this, an [Active Shape Model](https://en.wikipedia.org/wiki/Active_shape_model) (ASM) is constructed for each of the eight incisors. An Active Shape Model is a statistical model with the shape of a certain object that will be deformed during an iterative process to fit an instance of such an object in a different image.

### Input Landmarks
#### Image space
<p align="left">
<img src="data/Visualizations/Landmarks/landmarks1.png" width="215">
<img src="data/Visualizations/Landmarks/landmarks2.png" width="215">
<img src="data/Visualizations/Landmarks/landmarks3.png" width="215">
<img src="data/Visualizations/Landmarks/landmarks4.png" width="215">
</p>

### Active Shape Model construction
We construct an [Active Shape Model](https://en.wikipedia.org/wiki/Active_shape_model) (ASM) for each of the eight incisors. Possible alternatives are:
* One model for modelling each of the eight incisors. Based on the landmarks, we do not prefer this approach due to the clear differences between the shape of the upper and lower teeth and between the side and foreteeth.
* Multiple models for modelling one or more incisors.
* One model for modelling all eight incisors as a whole. Based on the landmarks, we do not prefer this approach due to the clear differences in distances between the upper and lower teeth.
* Multiple models for modelling one or more incisors as a whole. By constructing models for multiple incisors as a whole, possible correlations can be taken into account and exploited (e.g. neighbouring teeth influencing each other's position). This can be beneficiary during the fitting procedure. 
* Combinations of the aforementioned alternatives.

We chose to model each of the eight incisors separately because this is the most general approach. The difficulty with this approach is deciding on the initial solution for the iterative fitting process. Given 'well chosen' initial solutions, the fitting process should not encounter any problems due to this approach.

The first subsection describes the normalization of all the tooth shapes (described by the given landmarks) of the training samples for the same incisor tooth via a [Procrustes Analysis](https://en.wikipedia.org/wiki/Procrustes_analysis). Next, the shape variance is analysed via a [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). Note that the models, constructed in the following subsections, only describe the shape variance and not the appearance variance. The appearance will be taken into account during the construction of the fitting functions and the iterative fitting procedure.

#### Procrustes Analysis
Before generating an Active Shape Model, each tooth shape (described by its landmarks) of the set of training samples belonging to the same tooth needs to be aligned in a common coordinate system. First, we remove the translation component by centering each tooth shape's center of gravity at the origin. Second, we use a Procrustes Analys to align (scalar and rotation component) each tooth shape of the set of training samples belonging to same tooth, in such a way that the sum of the distances between the aligned tooth shape and the mean aligned tooth shape is minimized. [Cootes92], [Cootes00]

This is an iterative process. In each iteration step the mean aligned tooth shape is recalculated until the difference between two consecutive mean aligned tooth shapes in each dimension is smaller than a threshold value of 10<sup>-5</sup>. Initially, we used a larger value, but since convergence was reached after one or two iteration steps, we prefer a smaller value (more precision). With a threshold value of 10<sup>-5</sup>, convergence is reached after three iteration steps on average.

The resulting tooth models and tooth model landmarks are visualized below for each of the eight incisors together with the landmarks of 14 training samples in the same model (coordinate) space *(x', y')*. Note that the reference shape for aligning can be chosen arbitrarily with regard to the scaling and rotation component. We have chosen to align with each of the eight tooth shapes of training sample 1.

<p align="center">
<img src="data/Visualizations/PA/mean1-s.png" width="214">
<img src="data/Visualizations/PA/mean2-s.png" width="214">
<img src="data/Visualizations/PA/mean3-s.png" width="214">
<img src="data/Visualizations/PA/mean4-s.png" width="214">
</p>
<p align="center">
<img src="data/Visualizations/PA/mean5-s.png" width="214">
<img src="data/Visualizations/PA/mean6-s.png" width="214">
<img src="data/Visualizations/PA/mean7-s.png" width="214">
<img src="data/Visualizations/PA/mean8-s.png" width="214">
</p>

#### Principal Component Analysis
After the Procrustes Analysis, we have one model tooth for each of the eight incisor teeth in model space *(x', y')*. Each of these models is representative for the average shape of the corresponding incisor teeth. The aligned tooth shapes constitute a distribution in the *2n* dimensional space (with *n* the number of *2*-dimensional landmarks). We want to model this distribution.

We can for example use a basis for the *2n* dimensional space to represent each vector in this space as a *2n* dimensional coefficient vector (*2n* form factors). By applying a Principal Component Analysis (PCA), we can try to reduce the dimensionality by computing the principal axes (of which the magnitude is determined by the largest eigenvalues of the covariance matrix) of the distribution in order to express vectors approximately in this *2n* dimensional space with less form factors.

Based on the results of the Principal Component Analysis, it suffices to use for each of the eight incisors only 5, 6 or 7 form factors (versus *2n* = 40) to describe at least 98% of the variance of the landmark positions in the training set with regard to the corresponding mean model tooth. Alternatively, one can choose the form factors in such a way that each training sample can be represented with a certain accuracy.

By varying the form factors, we can vary the shape of the tooth. We need to determine upper and lower bounds for these variations in order to obtain plausible teeth. We allow +-3 standard deviations (*+-3 sqrt(lambda_i)*) from the mean model tooth. The effect of *-3 sqrt(lambda_i), -2 sqrt(lambda_i), -1 sqrt(lambda_i), 0, +1 sqrt(lambda_i), +2 sqrt(lambda_i), +3 sqrt(lambda_i)* deviations on each form factor as opposed to the mean tooth model for each of the eight incisors in model space *(x', y')*, is visualized below.

##### (reduced) model space tooth 1
<p align="left">
<img src="data/Visualizations/PCA/eig1-1.png" width="286">
<img src="data/Visualizations/PCA/eig1-2.png" width="286">
<img src="data/Visualizations/PCA/eig1-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig1-4.png" width="286">
<img src="data/Visualizations/PCA/eig1-5.png" width="286">
<img src="data/Visualizations/PCA/eig1-6.png" width="286">
</p>

##### (reduced) model space tooth 2
<p align="left">
<img src="data/Visualizations/PCA/eig2-1.png" width="286">
<img src="data/Visualizations/PCA/eig2-2.png" width="286">
<img src="data/Visualizations/PCA/eig2-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig2-4.png" width="286">
<img src="data/Visualizations/PCA/eig2-5.png" width="286">
<img src="data/Visualizations/PCA/eig2-6.png" width="286">
</p>

##### (reduced) model space tooth 3
<p align="left">
<img src="data/Visualizations/PCA/eig3-1.png" width="286">
<img src="data/Visualizations/PCA/eig3-2.png" width="286">
<img src="data/Visualizations/PCA/eig3-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig3-4.png" width="286">
<img src="data/Visualizations/PCA/eig3-5.png" width="286">
</p>

##### (reduced) model space tooth 4
<p align="left">
<img src="data/Visualizations/PCA/eig4-1.png" width="286">
<img src="data/Visualizations/PCA/eig4-2.png" width="286">
<img src="data/Visualizations/PCA/eig4-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig4-4.png" width="286">
<img src="data/Visualizations/PCA/eig4-5.png" width="286">
</p>

##### (reduced) model space tooth 5
<p align="left">
<img src="data/Visualizations/PCA/eig5-1.png" width="286">
<img src="data/Visualizations/PCA/eig5-2.png" width="286">
<img src="data/Visualizations/PCA/eig5-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig5-4.png" width="286">
<img src="data/Visualizations/PCA/eig5-5.png" width="286">
<img src="data/Visualizations/PCA/eig5-6.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig5-7.png" width="286">
</p>

##### (reduced) model space tooth 6
<p align="left">
<img src="data/Visualizations/PCA/eig6-1.png" width="286">
<img src="data/Visualizations/PCA/eig6-2.png" width="286">
<img src="data/Visualizations/PCA/eig6-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig6-4.png" width="286">
<img src="data/Visualizations/PCA/eig6-5.png" width="286">
<img src="data/Visualizations/PCA/eig6-6.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig6-7.png" width="286">
</p>

##### (reduced) model space tooth 7
<p align="left">
<img src="data/Visualizations/PCA/eig7-1.png" width="286">
<img src="data/Visualizations/PCA/eig7-2.png" width="286">
<img src="data/Visualizations/PCA/eig7-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig7-4.png" width="286">
<img src="data/Visualizations/PCA/eig7-5.png" width="286">
<img src="data/Visualizations/PCA/eig7-6.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig7-7.png" width="286">
</p>

##### (reduced) model space tooth 8
<p align="left">
<img src="data/Visualizations/PCA/eig8-1.png" width="286">
<img src="data/Visualizations/PCA/eig8-2.png" width="286">
<img src="data/Visualizations/PCA/eig8-3.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig8-4.png" width="286">
<img src="data/Visualizations/PCA/eig8-5.png" width="286">
<img src="data/Visualizations/PCA/eig8-6.png" width="286">
</p>
<p align="left">
<img src="data/Visualizations/PCA/eig8-7.png" width="286">
</p>

### Pre-process
The region of the dental radiographs containing the eight incisor teeth is relatively small compared to the full dental radiographs. Therefore, we first crop the dental radiographs based on the minimal and maximal landmark coordinates (and an extra safety border) of all the landmarks of the training samples. This approach is justified since the appliances for generating dental radiographs center the foreteeth, and since the appliances themselves are approximately standardized.

The given dental radiographs are inherently noisy data. We use `OpenCV`’s `fastNlMeansDenoising` function to remove Gaussian white noise from the (greyscale) images.

Furthermore, the intensity differences can be sometimes rather small, which can have a negative effect on the fitting functions operating on the intensity differences. We exploit the contrast to increase the intensity differences by using two different approaches: histogram equalization and (linear) contrast stretching.

Equalization maps a distribution with histogram *H(x)* on another distribution with a larger and more uniform distribution of intensity values. The mapping is done with the cumulative distribution function *H'(x)* of the histogram *H(x)*. [Fisher00] , [OpenCV11]

Contrast stretching aims to improve the contrast of an image by stretching the range of intensity values *[c, d]* (linearly) over a target range of intensity values *[a, b]=[0, 255]*. Outliers in the original range can have a negative impact on *[c, d]* resulting in a not representative scale. Therefore, we set *[c, d]* to the 5th and 95th percentile of the histogram, respectively. [Fisher00]

Some of the pre-processing techniques are illustrated below. In the remainder, we always use the 14 training samples after cropping, denoising and (linear) contrast stretching because these images give better results and convergence behavior. We have not investigated this further.

<p align="center">
<img src="data/Visualizations/Preproccess/O01.png" width="107">
<img src="data/Visualizations/Preproccess/D01.png" width="107">
<img src="data/Visualizations/Preproccess/EH01.png" width="107">
<img src="data/Visualizations/Preproccess/EHD01.png" width="107">
<img src="data/Visualizations/Preproccess/SC01.png" width="107">
<img src="data/Visualizations/Preproccess/SCD01.png" width="107">
</p>

1. Cropping (Original Greyscale)
2. Cropping -> Denoising
3. Cropping -> Histogram Equalization
4. Cropping -> Denoising -> Histogram Equalization
5. Cropping -> Linear Contrast Stretching
6. Cropping -> Denoising -> Linear Contrast Stretching

### Fitting
The first subsection describes the selection and construction of the fitting functions which will be used in the fitting procedures. The following subsections explain the iterative single- and multi-resolution ASM fitting procedures, and the procedures for the manual and automatic generation of an initial input solution for these fitting procedures.

#### Construction of the fitting functions
#### Single-resolution Active Shape Model’s fitting procedure
#### Multi-resolution Active Shape Model’s fitting procedure
#### Initialisation
##### Manuel initialization
##### Automatic initialization
