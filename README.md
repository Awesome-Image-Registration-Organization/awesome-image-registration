# image-registration-resources
image registration related books, papers, videos, and toolboxes 

[![Stars](https://img.shields.io/github/stars/youngfish42/image-registration-resources.svg?color=orange)](https://github.com/youngfish42/image-registration-resources/stargazers) 
[![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86%E6%8C%87%E5%8C%97-blue)](https://zhuanlan.zhihu.com/Image-Registration) 
[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
[![License](https://img.shields.io/github/license/youngfish42/image-registration-resources.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) 


Many thanks to [**yzhao062**](https://github.com/yzhao062/anomaly-detection-resources/commits?author=yzhao062) [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources), and I modeled his style to make this depository. 

---

[**Image registration**](https://en.wikipedia.org/wiki/Image_registration) is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, data from different sensors, times, depths, or viewpoints.

It is used in computer vision, medical imaging, military automatic target recognition, and compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from these different measurements. 



This repository collects:
- Books & Academic Papers 
- Online Courses and Videos
- Datasets
- Open-source and Commercial Libraries/Toolkits
- Key Conferences & Journals


**More items will be added to the repository**.
Please feel free to suggest other key resources by opening an issue report,
submitting a pull request, or dropping me an email @ (im.young@foxmail.com).
Enjoy reading!

---


## 1. Books & Tutorials    
### 1.1. Books

#### natural image

[Multiple view geometry in computer vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) by Richard Hartley and Andrew Zisserman, 2004: Mathematic and geometric basis for 2D-2D and 2D-3D registration. A **must-read** for people in the field of registration. [[E-book\]](http://cvrs.whu.edu.cn/downloads/ebooks/Multiple View Geometry in Computer Vision (Second Edition).pdf)

[Computer Vision: A Modern Approach](http://www.informit.com/store/computer-vision-a-modern-approach-9780136085928) by David A. Forsyth, Jean Ponce:  for upper-division undergraduate- and  graduate-level courses in computer vision found in departments of  Computer Science, Computer Engineering and Electrical Engineering.

[Algebra, Topology, Differential Calculus, and Optimization Theory For Computer Science and Engineering](https://www.cis.upenn.edu/~jean/gbooks/geomath.html) By **Jean Gallier and Jocelyn Quaintance**. The latest book from upenn about the algebra and optimization theory.

[Three-Dimensional Computer vision-A Geometric Viewpoint](https://mitpress.mit.edu/books/three-dimensional-computer-vision)  Classical 3D computer vision textbook.

[An invitation to 3D vision](https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf) a self-contained introduction to the geometry of three-dimensional (3-D) vision.

#### Point Cloud

[14 lectures on visual SLAM](https://github.com/gaoxiang12/slambook) By Xiang Gao and Tao Zhang and Yi Liu and Qinrui Yan.  **SLAM**  [[github\]](https://github.com/gaoxiang12/slambook) [[Videos\]](https://space.bilibili.com/38737757)

[点云数据配准及曲面细分技术](https://baike.baidu.com/item/点云数据配准及曲面细分技术/10225974) by 薛耀红, 赵建平, 蒋振刚, 等   书籍内容比较过时，仅适合零基础读者阅读。推荐自行查找相关博客学习。

#### Remote Sensing

[2-D and 3-D Image Registration: For Medical, Remote Sensing, and Industrial Applications](www.researchgate.net/profile/Rachakonda_Poojitha/post/How_to_reconstruct_a_3D_image_from_two_2D_images_of_the_same_scene_taken_from_the_same_camera/attachment/59d61d9d6cda7b8083a16a8f/AS%3A271832186327046%401441821251591/download/2-D+and+3-D+Image+Registration+for+Medical%2C+Remote+Sensing%2C+and+Industrial+Applications.pdf) by  A. A. Goshtasby, 2005.  



### 1.2. Tutorials

#### Natural image



#### Medical Image

- MICCAI2019: [learn2reg](https://github.com/learn2reg/tutorials2019) [PDF](https://github.com/learn2reg/tutorials2019/blob/master/slides)
> Medical image registration has been a cornerstone in the research fields of medical image computing and computer assisted intervention, responsible for many clinical applications. Whilst machine learning methods have long been important in developing pairwise algorithms, recently proposed deep-learning-based frameworks directly infer displacement fields without iterative optimisation for unseen image pairs, using neural networks trained from large population data. These novel approaches promise to tackle several most challenging aspects previously faced by classical pairwise methods, such as high computational cost, robustness for generalisation and lack of inter-modality similarity measures. Output from several international research groups working in this area include award-winning conference presentations, high-impact journal publications, well-received open-source implementations and industrial-partnered translational projects, generating significant interests to all levels of world-wide researchers. Accessing to the experience and expertise in this inherently multidisciplinary topic can be beneficial to many in our community, especially for the next generation of young scientists, engineers and clinicians who often have only been exposed to a subset of these methodologies and applications. We organise a tutorial including both theoretical and practical sessions, inviting expert lectures and tutoring coding for real-world examples. Three hands-on sessions guiding participants to understand and implement published algorithms using clinical imaging data. This aims to provide an opportunity for the participants to bridge the gap between expertises in medical image registration and deep learning, as well as to start a forum to discuss knowhows, challenges and future opportunities in this area.
- [kaggle:2016] [Image registration, the R way, (almost) from scratch](https://www.kaggle.com/vicensgaitan/image-registration-the-r-way)
> There are some packages in R for image manipulation and after some test I select “imager” , based on the CImg C++, fast and providing several image processing tools.
- [kaggle:2018] [X-Ray Patient Scan Registration](https://www.kaggle.com/kmader/x-ray-patient-scan-registration)
> SimpleITK, ITK, scipy, OpenCV, Tensorflow and PyTorch all offer tools for registering images, we explore a few here to see how well they work when applied to the fairly tricky problem of registering from the same person at different time and disease points.



#### Remote Sensing



#### Point Cloud

- [点云配准算法说明与流程介绍](https://blog.csdn.net/Ha_ku/article/details/79755623)

- [点云配准算法介绍与比较](https://blog.csdn.net/weixin_43236944/article/details/88188532)

- [机器学习方法处理三维点云](https://blog.csdn.net/u014636245/article/details/82755966)

- [一个例子详细介绍点云配准的过程](https://www.zhihu.com/question/34170804/answer/121533317)





## 2. Courses/Seminars/Videos

## 3. Toolbox 

### Natural image



### Medical Image

[c++] [ITK](https://itk.org/):   **Insight Toolkit (ITK)**  an open-source, cross-platform system  that provides developers  with an extensive suite of software tools for image  analysis.  Developed through extreme  programming methodologies, ITK employs  leading-edge algorithms for registering  and segmenting multidimensional data.  

[c++] [Python] [Java] [SimpleITK](http://www.simpleitk.org/): A simplified layer built on top of ITK.

[c++] [ANTs](http://stnava.github.io/ANTs/): Advanced normalization tools for brain and image analysis.  Image registration with variable transformations (elastic,  diffeomorphic, diffeomorphisms, unbiased) and similarity metrics  (landmarks, cross-correlation, mutual information, etc). Image  segmentation with priors & nonparametric, multivariate models.   

[c++] [Elastix](http://elastix.isi.uu.nl/):  open source software, based on the well-known [Insight Segmentation and Registration Toolkit](http://www.itk.org) (ITK). The software consists of a collection of algorithms that are commonly used to solve (medical) image registration problems.  

[C++] [Python] [Java] [R] [Ruby] [Lua] [Tcl] [C#] [SimpleElastix](http://simpleelastix.github.io/): a medical image registration library that makes  state-of-the-art image registration really easy to do in languages like  Python, Java and R. 

[3D slicer](https://www.slicer.org/) :  an open source software platform for  medical image informatics, image processing, and three-dimensional  visualization. Built over two decades through support from the  National Institutes of Health and a worldwide developer community, Slicer brings free, powerful cross-platform processing tools to  physicians, researchers, and the general public.  



**Github repository for medical image registration**

[Keras] [VoxelMorph](https://github.com/voxelmorph/voxelmorph)

[Keras] [FAIM](https://gihttps://github.com/dykuang/Medical-image-registrationthub.com/dykuang/Medical-image-registration)

[Tensorflow] [Weakly-supervised CNN](https://github.com/YipengHu/label-reg) 

[Tensorflow] [RegNet3D](https://github.com/hsokooti/RegNet) 

[Tensorflow] [Recursive-Cascaded-Networks](https://github.com/microsoft/Recursive-Cascaded-Networks)  

[Pytorch] [Probabilistic Dense Displacement Network](https://github.com/multimodallearning/pdd_net)

[Pytorch] [Linear and Deformable Image Registration](https://github.com/shreshth211/image-registration-cnn)

[Pytorch] [Inverse-Consistent Deep Networks](https://github.com/zhangjun001/ICNet) 

[Pytorch] [Non-parametric image registration](https://github.com/uncbiag/registration) 

[Pytorch] [One Shot Deformable Medical Image Registration](https://github.com/ToFec/OneShotImageRegistration)

[Pytorch] [Image-and-Spatial Transformer Networks](https://github.com/biomedia-mira/istn)



### Remote Sensing



### Point Cloud



## 4. Datasets & Competitions
### 4.1. Datasets
- [kaggle:2018] [ct-scans-before-and-after](https://www.kaggle.com/kmader/ct-scans-before-and-after)
> The dataset is supposed to make it easier to see and explore different registration techniques in particular [VoxelMorph](https://github.com/voxelmorph/voxelmorph)

### 4.2. Competitions


#### [**All Challenges**](https://grand-challenge.org/challenges/)

##### 2019 

[CuRIOUS:2019](https://curious2019.grand-challenge.org/) | [Official solution](https://arxiv.org/ftp/arxiv/papers/1904/1904.10535.pdf)
> 1 Register pre-operative MRI to iUS before tumor resection  
> 2 Register iUS after tumor resection to iUS before tumor resection  

[ANHIR:2019](https://anhir.grand-challenge.org/) | [Official solution](https://www.researchgate.net/publication/332428245_Automatic_Non-rigid_Histological_Image_Registration_challenge)
> IEEE International Symposium on Biomedical Imaging (ISBI) 2019  
> High-resolution (up to 40x magnification) whole-slide images of tissues (lesions, lung-lobes, mammary-glands) were acquired - the original size of our images is up to 100k x 200k pixels. The acquired images are organized in sets of consecutive sections where each slice was stained with a different dye and any two images within a set can be meaningfully registered.

##### 2018 

[iChallenges ](https://ichallenges.grand-challenge.org/) 

[Continuous Registration Challenge](https://continuousregistration.grand-challenge.org/) 

[Multi-shell Diffusion MRI Harmonisation Challenge 2018 (MUSHAC)](https://projects.iq.harvard.edu/cdmri2018/challenge)

##### 2010 

[EMPIRE10](http://empire10.isi.uu.nl/)


## 5. Papers
### 5.1. Overview & Survey Papers
### 5.2. Key Algorithms
### 5.3. 


## 6. Key Conferences/Workshops/Journals
### 6.1. Conferences & Workshops
### 6.2. Journals


