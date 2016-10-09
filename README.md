# Incisor Segmentation

Course Computer Vision: a model-based procedure capable of segmenting the incisors in panoramic dental radiographs using an Active Shape Model (ASM). 

**Team**:
* [Matthias Moulin](https://github.com/matt77hias) (Computer Science)
* [Milan Samyn](https://github.com/MilanSamyn) (Computer Science)

**Academic Year**: 2014-2015 (2nd semester - 2nd Master of Science in Engineering: Computer Science)

## About

### Input Landmarks
#### Image space
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks1.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks2.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks3.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Landmarks/landmarks4.png" width="215"></p>

### Procrustes Analysis (PA)
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean1-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean2-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean3-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean4-s.png" width="215"></p>
<p align="left"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean5-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean6-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean7-s.png" width="215"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/PA/mean8-s.png" width="215"></p>

### Principal Component Analysis (PCA)
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
<p align="center"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/D01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/EH01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/EHD01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/SC01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/SCD01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/S01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/SD01.png" width="107"><img src="https://github.com/matt77hias/IncisorSegmentation/blob/master/data/Visualizations/Preproccess/O01.png" width="107"></p>

### Fitting functions
