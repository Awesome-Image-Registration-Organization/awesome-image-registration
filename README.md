# image-registration-resources

image registration related books, papers, videos, and toolboxes 

[![Stars](https://img.shields.io/github/stars/youngfish42/image-registration-resources.svg?color=orange)](https://github.com/youngfish42/image-registration-resources/stargazers) 
[![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86%E6%8C%87%E5%8C%97-blue)](https://zhuanlan.zhihu.com/Image-Registration) 
[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
[![License](https://img.shields.io/github/license/youngfish42/image-registration-resources.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) 

Many thanks to [**yzhao062**](https://github.com/yzhao062/anomaly-detection-resources/commits?author=yzhao062) [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources). I followed his style to make this depository. 







---

[**Image registration**](https://en.wikipedia.org/wiki/Image_registration) is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, and from different sensors, times, depths, or viewpoints.

It is used in computer vision, medical imaging, military automatic target recognition, compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from different measurements. 



This repository collects:

- Books & Academic Papers 
- On-line Courses and Videos
- Datasets
- Open-source and Commercial Libraries/Toolkits
- Key Conferences & Journals



[toc]




**More items will be added to the repository**.
Please feel free to suggest other key resources by opening an issue report,
submitting a pull request, or dropping me an email @ (im.young@foxmail.com).
Enjoy reading!

---


## 1. Books & Tutorials    

### 1.1. Books

#### Natural image

[Multiple view geometry in computer vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) by Richard Hartley and Andrew Zisserman, 2004: Mathematic and geometric basis for 2D-2D and 2D-3D registration. A **must-read** for people in the field of registration. [E-book](http://cvrs.whu.edu.cn/downloads/ebooks/Multiple%20View%20Geometry%20in%20Computer%20Vision%20(Second%20Edition).pdf)

[Computer Vision: A Modern Approach](http://www.informit.com/store/computer-vision-a-modern-approach-9780136085928) by David A. Forsyth, Jean Ponce:  for upper-division undergraduate- and  graduate-level courses in computer vision found in departments of  Computer Science, Computer Engineering and Electrical Engineering.

[Algebra, Topology, Differential Calculus, and Optimization Theory For Computer Science and Engineering](https://www.cis.upenn.edu/~jean/gbooks/geomath.html) by Jean Gallier and Jocelyn Quaintance. The latest book from upenn about the algebra and optimization theory.

[Three-Dimensional Computer vision-A Geometric Viewpoint](https://mitpress.mit.edu/books/three-dimensional-computer-vision)  Classical 3D computer vision textbook.

[An invitation to 3D vision](https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf) a self-contained introduction to the geometry of three-dimensional (3-D) vision.

#### Medical Image

Zhenhuan Zhou, et.al: [ **A software guide for medical image segmentation and registration algorithm. 医学图像分割与配准(ITK实现分册)**](https://vdisk.weibo.com/s/FQyto0RT-heb) 
Part Ⅱ introduces the most basic network and architecture of medical registration algorithms **(Chinese Version)**.

[2-D and 3-D Image Registration for Medical, Remote Sensing, and Industrial Applications](http://www.researchgate.net/profile/Rachakonda_Poojitha/post/How_to_reconstruct_a_3D_image_from_two_2D_images_of_the_same_scene_taken_from_the_same_camera/attachment/59d61d9d6cda7b8083a16a8f/AS%3A271832186327046%401441821251591/download/2-D+and+3-D+Image+Registration+for+Medical%2C+Remote+Sensing%2C+and+Industrial+Applications.pdf) by A. Ardeshir Goshtasby

[医学图像配准技术与应用](https://book.douban.com/subject/26411955/) by 吕晓琪    

[Intensity-based 2D-3D Medical Image Registration](https://blackwells.co.uk/bookshop/product/9783639119541) by Russakoff, Daniel

[Biomedical Image Registration](https://www.springer.com/gb/book/9783642143656) by Fischer, Dawant, Lorenz

[Medical Image Registration](https://wordery.com/medical-image-registration-joseph-v-hajnal-9780849300646) by  Hajnal, Joseph V.

[Deep Learning for Medical Image Analysis](https://www.elsevier.com/books/deep-learning-for-medical-image-analysis/zhou/978-0-12-810408-8) (part IV)



#### Point Cloud

[14 lectures on visual SLAM](https://github.com/gaoxiang12/slambook) By Xiang Gao and Tao Zhang and Yi Liu and Qinrui Yan.  **视觉SLAM十四讲**  视觉配准方向较易懂的入门教材。通俗讲述视觉匹配的物理模型， 数学几何基础，优化过程等。 新手必读。 [[github\]](https://github.com/gaoxiang12/slambook) [[Videos\]](https://space.bilibili.com/38737757)

[点云数据配准及曲面细分技术](https://baike.baidu.com/item/点云数据配准及曲面细分技术/10225974) by 薛耀红, 赵建平, 蒋振刚, 等   书籍内容比较过时，仅适合零基础读者阅读。推荐自行查找相关博客学习。

#### Remote Sensing


[Image Registration for Remote Sensing](https://www.amazon.com/Registration-Remote-Sensing-Jacqueline-Moigne-ebook/dp/B005252MNG/)

[2-D and 3-D Image Registration: For Medical, Remote Sensing, and Industrial Applications](www.researchgate.net/profile/Rachakonda_Poojitha/post/How_to_reconstruct_a_3D_image_from_two_2D_images_of_the_same_scene_taken_from_the_same_camera/attachment/59d61d9d6cda7b8083a16a8f/AS%3A271832186327046%401441821251591/download/2-D+and+3-D+Image+Registration+for+Medical%2C+Remote+Sensing%2C+and+Industrial+Applications.pdf) by  A. A. Goshtasby, 2005.  

[航空遥感图像配准技术](https://book.douban.com/subject/26711943/)

[基于特征的光学与SAR遥感图像配准](https://item.jd.com/12099246.html)

[基于特征的航空遥感图像配准及部件检测技术](https://item.jd.com/12576983.html)

[Introduction to Remote Sensing](https://www.amazon.com/Introduction-Remote-Sensing-Fifth-Campbell/dp/160918176X/)

[Remote Sensing and Image Interpretation](https://www.amazon.com/Remote-Sensing-Interpretation-Thomas-Lillesand/dp/111834328X/)

[Remote Sensing: Models and Methods for Image Processing](https://www.amazon.com/Remote-Sensing-Models-Methods-Processing/dp/0123694078)
### 1.2. Tutorials

#### Natural image



#### Medical Image

- [**Medical Image Registration**](https://github.com/natandrade/Tutorial-Medical-Image-Registration) 

- [MICCAI2019] [**learn2reg**](https://github.com/learn2reg/tutorials2019) [PDF](https://github.com/learn2reg/tutorials2019/blob/master/slides)

> Big thanks to [Yipeng Hu]( https://github.com/YipengHu ) organizing the excellent tutorial.
>
> **Description:**
>
> Medical image registration has been a cornerstone in the research fields of medical image computing and computer assisted intervention, responsible for many clinical applications. Whilst machine learning methods have long been important in developing pairwise algorithms, recently proposed deep-learning-based frameworks directly infer displacement fields without iterative optimization for unseen image pairs, using neural networks trained from large population data. These novel approaches promise to tackle several most challenging aspects previously faced by classical pairwise methods, such as high computational cost, robustness for generalization and lack of inter-modality similarity measures. 
>
> Output from several international research groups working in this area include award-winning conference presentations, high-impact journal publications, well-received open-source implementations and industrial-partnered translational projects, generating significant interests to all levels of world-wide researchers. Accessing to the experience and expertise in this inherently multidisciplinary topic can be beneficial to many in our community, especially for the next generation of young scientists, engineers and clinicians who often have only been exposed to a subset of these methodologies and applications. 
>
> We organize a tutorial including both theoretical and practical sessions, inviting expert lectures and tutoring coding for real-world examples. Three hands-on sessions guiding participants to understand and implement published algorithms using clinical imaging data. This aims to provide an opportunity for the participants to bridge the gap between expertises in medical image registration and deep learning, as well as to start a forum to discuss know-hows, challenges and future opportunities in this area.

- [kaggle:2016] [**Image registration, the R way, (almost) from scratch**](https://www.kaggle.com/vicensgaitan/image-registration-the-r-way)

> There are some packages in R for image manipulation and after some test I select “imager” , based on the CImg C++, fast and providing several image processing tools.

- [kaggle:2018] [**X-Ray Patient Scan Registration**](https://www.kaggle.com/kmader/x-ray-patient-scan-registration)

> SimpleITK, ITK, scipy, OpenCV, Tensorflow and PyTorch all offer tools for registering images, we explore a few here to see how well they work when applied to the fairly tricky problem of registering from the same person at different time and disease points.

- [MICCAI2019] [**Autograd Image Registration Laboratory**](https://github.com/airlab-unibas/MICCAITutorial2019)

- [MIT] [**HST.582J**](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-582j-biomedical-signal-and-image-processing-spring-2007/)  Biomedical Signal and Image Processing [PDF](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-582j-biomedical-signal-and-image-processing-spring-2007/lecture-notes/l16_reg1.pdf) 





#### Remote Sensing
- [Image Alignment and Stitching: A Tutorial](http://www.cs.toronto.edu/~kyros/courses/2530/papers/Lecture-14/Szeliski2006.pdf)

- [Image Registration for Remote Sensing](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120008278.pdf)

- [Image Stitching](https://www.zhihu.com/question/34535199/answer/135169187)

- [The Remote Sensing Tutorial 1](https://www.ucl.ac.uk/EarthSci/people/lidunka/GEOL2014/Geophysics%2010%20-Remote%20sensing/Remote%20Sensing%20Tutorial%20Overview.htm)

- [The Remote Sensing Tutorial 2](https://www.nrcan.gc.ca/maps-tools-publications/satellite-imagery-air-photos/tutorial-fundamentals-remote-sensing/9309)
  
#### Point Cloud

- [点云配准算法说明与流程介绍](https://blog.csdn.net/Ha_ku/article/details/79755623)

- [点云配准算法介绍与比较](https://blog.csdn.net/weixin_43236944/article/details/88188532)

- [机器学习方法处理三维点云](https://blog.csdn.net/u014636245/article/details/82755966)

- [一个例子详细介绍点云配准的过程](https://www.zhihu.com/question/34170804/answer/121533317)



### 1.3. Blogs

#### [图像配准指北](https://zhuanlan.zhihu.com/Image-Registration)

> [图像配准综述](https://zhuanlan.zhihu.com/p/80985475) 
>
> [基于深度学习的医学图像配准综述](https://zhuanlan.zhihu.com/p/70820773) 
>
> [基于深度学习和图像引导的医学图像配准](https://zhuanlan.zhihu.com/p/82423947) 
>
> [图像配准：从SIFT到深度学习](https://zhuanlan.zhihu.com/p/75784915) 
>
> [点云配准综述](https://zhuanlan.zhihu.com/p/91275450) 
>
> **图像配准会议介绍@** [MICCAI2019](https://zhuanlan.zhihu.com/p/87781312) / [CVPR2019](https://zhuanlan.zhihu.com/p/78798607) / [ICCV2019](https://zhuanlan.zhihu.com/p/80529725) / [NeurIPS2019](https://zhuanlan.zhihu.com/p/81658522)

[Image Registration: From SIFT to Deep Learning]( https://blog.sicara.com/image-registration-sift-deep-learning-3c794d794b7a)



#### 点云配准 (！该部分需要重新编辑)

[点云配准算法的说明与流程介绍](https://blog.csdn.net/Ha_ku/article/details/797556232)

[几种点云配准算法的方法的介绍与比较](https://blog.csdn.net/weixin_43236944/article/details/881885323)

[三维点云用机器学习的方法进行处理](https://blog.csdn.net/u014636245/article/details/827559664)

[一个例子详细介绍了点云配准的过程](https://www.zhihu.com/question/34170804)

---

## 2. Courses/Seminars/Videos

### Courses

[**16-822: Geometry-based Methods in Vision**](http://www.cs.cmu.edu/~hebert/geom.html)

[VALSE 2018] [Talk: 2017以来的2D to 3D](https://zhuanlan.zhihu.com/p/38611920) by 吴毅红



### Workshops

WBIR - International Workshop on Biomedical Image Registration

> [WBIR2018,](https://wbir2018.nl/index.html) Leiden, Netherlands
> WBIR2016, Las Vegas NV 
> WBIR2014, London, UK  



### Seminars



### Videos

- [Definition and Introduction to Image Registration Pre Processing Overview](https://www.youtube.com/watch?v=sGNFmAGqpZ8)

- [仿射变换与图像配准](https://www.bilibili.com/video/av52733294)（科普性视频， 比较简陋）

#### Remote Sensing
- [Registration of images of different modalities in Remote Sensing](https://youtu.be/9pPwNN-7oWU)



---

## 3. Toolbox 

### Natural image

[C++]  [Python] [OpenCV](https://opencv.org/): OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to  provide a common infrastructure for computer vision applications and to  accelerate the use of machine perception in the commercial products.

[C++] [PCL: Point Cloud Library](http://pointclouds.org/). The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing.

[C++] [Ceres Solver](http://ceres-solver.org/index.html): Ceres Solver is an open source C++ library for modeling and solving  large, complicated optimization problems. It can be used to solve  Non-linear Least Squares problems with bounds constraints and general  unconstrained optimization problems.

[C++] [Open3D](http://www.open3d.org/): Open3D is an open-source library that supports rapid development of  software that deals with 3D data. The Open3D frontend exposes a set of  carefully selected data structures and algorithms in both C++ and  Python. The backend is highly optimized and is set up for  parallelization.

### Medical Image

[c++] [**ITK**](https://itk.org/): Segmentation & Registration Toolkit

An open-source, cross-platform system  that provides developers  with an extensive suite of software tools for image  analysis.  Developed through extreme  programming methodologies. ITK employs  leading-edge algorithms for registering  and segmenting multidimensional data.  

[c++] [Python] [Java] [**SimpleITK**](http://www.simpleitk.org/): a simplified layer built on top of ITK.

[c++] [**ANTs**](http://stnava.github.io/ANTs/): Advanced Normalization Tools.  

Image registration with variable transformations (elastic,  diffeomorphic, diffeomorphisms, unbiased) and similarity metrics  (landmarks, cross-correlation, mutual information, etc.). Image  segmentation with priors & nonparametric, multivariate models.   

[c++] [**Elastix**](http://elastix.isi.uu.nl/):  open source software, based on the well-known [ITK](http://www.itk.org) . 

The software consists of a collection of algorithms that are commonly used to solve (medical) image registration problems.  [**[manual]**](http://elastix.isi.uu.nl/download/elastix_manual_v4.8.pdf) 

[C++] [Python] [Java] [R] [Ruby] [Lua] [Tcl] [C#] [**SimpleElastix**](http://simpleelastix.github.io/): a medical image registration library that makes  state-of-the-art image registration really easy to do in languages like  Python, Java and R. 

[**3D slicer**](https://www.slicer.org/) :  an open source software platform for  medical image informatics, image processing, and three-dimensional  visualization. Built over two decades through support from the  National Institutes of Health and a worldwide developer community, Slicer brings free, powerful cross-platform processing tools to  physicians, researchers, and the general public.  



**Github repository for deep learning medical image registration**:

 [Keras] [**VoxelMorph**](https://github.com/voxelmorph/voxelmorph)

 [Keras] [**FAIM**]( https://github.com/dykuang/Medical-image-registration )

 [Tensorflow] [**Weakly-supervised CNN**](https://github.com/YipengHu/label-reg)

 [Tensorflow] [**RegNet3D** ](https://github.com/hsokooti/RegNet)

 [Tensorflow] [**Recursive-Cascaded-Networks**](https://github.com/microsoft/Recursive-Cascaded-Networks) 

 [Pytorch] [**Probabilistic Dense Displacement Network**](https://github.com/multimodallearning/pdd_net)

 [Pytorch] [**Linear and Deformable Image Registration**](https://github.com/shreshth211/image-registration-cnn)

 [Pytorch] [**Inverse-Consistent Deep Networks**](https://github.com/zhangjun001/ICNet) 

 [Pytorch] [**Non-parametric image registration**](https://github.com/uncbiag/registration) 

 [Pytorch] [**One Shot Deformable Medical Image Registration**](https://github.com/ToFec/OneShotImageRegistration)

 [Pytorch] [**Image-and-Spatial Transformer Networks**](https://github.com/biomedia-mira/istn)



### Remote Sensing

[C++] [OTB](https://github.com/orfeotoolbox/OTB): Orfeo ToolBox (OTB) is an open-source project for state-of-the-art remote sensing. Built on the shoulders of the open-source geospatial community, it can process high resolution optical, multispectral and radar images at the terabyte scale. A wide variety of applications are available: from ortho-rectification or pansharpening, all the way to classification, SAR processing, and much more!

[C++] [Python] [OpenCV](https://github.com/opencv/opencv): OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to  provide a common infrastructure for computer vision applications and to  accelerate the use of machine perception in the commercial products.Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.

[C++] [ITK](https://itk.org/):   **Insight Toolkit (ITK)**  an open-source, cross-platform system  that provides developers  with an extensive suite of software tools for image  analysis.  Developed through extreme  programming methodologies, ITK employs  leading-edge algorithms for registering  and segmenting multidimensional data.

[Python] [Spectral Python (SPy)](https://github.com/spectralpython/spectral): Spectral Python (SPy) is a pure Python module for processing hyperspectral image data (imaging spectroscopy data). It has functions for reading, displaying, manipulating, and classifying hyperspectral imagery. 

**Post Processing Tools**

[C++] [enblend](https://sourceforge.net/projects/enblend/): Enblend blends away the seams in a panoramic image mosaic using a multi-resolution spline. Enfuse merges different exposures of the same scene to produce an image that looks much like a tone-mapped image.

[C++] [maxflow](https://pub.ist.ac.at/~vnk/software.html): An implementation of the maxflow algorithm which can be used to detect the optimal seamline.

[C++] [Matlab] [gco-v3.0](https://github.com/nsubtil/gco-v3.0): Multi-label optimization library by Olga Veksler and Andrew Delong.

**Source Code**

[APAP](https://cs.adelaide.edu.au/~tjchin/apap/)

[AANAP](https://github.com/YaqiLYU/AANAP)

[NISwGSP](https://github.com/firdauslubis88/NISwGSP)

[SPHP](https://www.cmlab.csie.ntu.edu.tw/~frank/SPH/cvpr14_SPHP_code.tar)

[Parallax-tolerant image stitching](https://github.com/gain2217/Robust_Elastic_Warping)



### Point Cloud

#### MeshLab

> 简介：是一款开源、可移植和可扩展的三维几何处理系统。主要用于处理和编辑3D三角网格，它提供了一组用于编辑、清理、修复、检查、渲染、纹理化和转换网格的工具。提供了处理由3D数字化工具/设备生成的原始数据以及3D打印功能，功能全面而且丰富。MeshLab支持多数市面上常见的操作系统，包括Windows、Linux及Mac OS X，支持输入/输出的文件格式有：STL 、OBJ 、 VRML2.0、U3D、X3D、COLLADA
> MeshLab可用于各种学术和研究环境，如微生物学、文化遗产及表面重建等。

#### ICP开源库

[SLAM6D](http://slam6d.sourceforge.net/)

[Libicp](http://www.cvlibs.net/software/libicp/)

[libpointmatcher](https://github.com/ethz-asl/libpointmatcher)

[g-icp](https://github.com/avsegal/gicp)

[n-icp](http://jacoposerafin.com/nicp/)

---

## 4. Datasets & Competitions

### 4.1. Datasets

#### Natural image

[**Indoor LiDAR-RGBD Scan Dataset**](http://redwood-data.org/indoor_lidar_rgbd/index.html)

[**ETH3D SLAM & Stereo Benchmarks**](https://www.eth3d.net/)

[**EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

[**ViViD : Vision for Visibility Dataset**](https://sites.google.com/view/dgbicra2019-vivid)

[**Apolloscape: Scene Parsing**]( http://apolloscape.auto/scene.html)

[**KITTI Visual Odometry dataset**](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

[**NCLT Dataset**](http://robots.engin.umich.edu/nclt/)

[**Oxford Robotcar Dataset**](https://robotcar-dataset.robots.ox.ac.uk/)



#### Medical Image 

|                           Dataset                            | Number | Modality |  Region  |     Format      |
| :----------------------------------------------------------: | :----: | :------: | :------: | :-------------: |
|             [DIRLAB](https://www.dir-lab.com/ )              |   10   |  4D  CT  |   Lung   |      .img       |
|                          [LPBA40]()                          |   40   | 3D  MRI  | T1 Brain | .img+.hdr  .nii |
|        [IBSR18](https://www.nitrc.org/projects/ibsr/)        |   18   | 3D  MRI  | T1 Brain |    .img+.hdr    |
|             [EMPIRE](http://empire10.isi.uu.nl/)             |   30   |  4D CT   |   Lung   |    .mhd+.raw    |
|                          [LiTS]( )                           |  131   |  3D CT   |  Liver   |      .nii       |
| [CT-scans-before-and-after](https://www.kaggle.com/kmader/ct-scans-before-and-after) |        |          |          |                 |
|            [Openi](https://openi.nlm.nih.gov/faq)            |        |  X-ray   |          |                 |
| [POPI](https://www.creatis.insa-lyon.fr/rio/popi-model?action=show&redirect=popi) |   6    |  4D CT   |          |                 |
|            [NLST](https://cdas.cancer.gov/nlst/)             |        |    CT    |   Lung   |                 |
|  [ADNI](http://adni.loni.usc.edu/data-samples/access-data/)  |        |  3D MRI  |  Brain   |                 |
|            [OASIS](http://www.oasis-brains.org/)             |        |  3D MRI  |  Brain   |                 |
| [ABIDE](http://preprocessed-connectomes-project.org/abide/)  |        |  3D MRI  |  Brain   |                 |
| [ADHD200](http://neurobureau.projects.nitrc.org/ADHD200/Introduction.html) |        |          |          |                 |
|    [CUMC12](https://www.synapse.org/#!Synapse:syn3207203)    |   12   |  3D MRI  |  Brain   |    .img+.hdr    |
|    [MGH10](https://www.synapse.org/#!Synapse:syn3207203)     |   10   |  3D MRI  |  Brain   |    .img+.hdr    |
|    [FIRE](https://www.ics.forth.gr/cvrl/fire/)     |   134   |  2D fundus |  Retina   |    .jpg    |

#### Remote Sensing 

[**ISPRS Benchmarks**](https://www.isprs.org/education/benchmarks.aspx)

[**HPatches**](https://github.com/hpatches/hpatches-dataset): The HPatches dataset was used as the basis for the local descriptor evaluation challenge that was presented in the Local Features: State of the Art, Open Problems and Performance Evaluation workshop during ECCV 2016.

[**The Zurich Urban Micro Aerial Vehicle Dataset**](http://rpg.ifi.uzh.ch/zurichmavdataset.html)

[**Zurich Summer Dataset**](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset)

[**Inria Aerial Image Labeling DataSet**](https://project.inria.fr/aerialimagelabeling/)

[**LANDSAT**](https://github.com/olivierhagolle/LANDSAT-Download)

[**NWPU-RESISC45**](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)

[**DOTA**](http://captain.whu.edu.cn/DOTAweb/index.html)

[**MUUFLGulfport**](https://github.com/GatorSense/MUUFLGulfport)



#### Point Cloud(！该部分需要重新编辑)

**The Stanford 3D Scanning Repository**（斯坦福大学的3d扫描存储库）

http://graphics.stanford.edu/data/3Dscanrep/

这应该是做点云数据最初大家用最多的数据集，其中包含最开始做配准的Bunny、Happy Buddha、Dragon等模型。



**Shapenet**

ShapeNet是一个丰富标注的大规模点云数据集，其中包含了55中常见的物品类别和513000个三维模型。

The KITTI Vision Benchmark Suite

链接：http://www.cvlibs.net/datasets/kitti/

这个数据集来自德国卡尔斯鲁厄理工学院的一个项目，其中包含了利用KIT的无人车平台采集的大量城市环境的点云数据集（KITTI），这个数据集不仅有雷达、图像、GPS、INS的数据，而且有经过人工标记的分割跟踪结果，可以用来客观的评价大范围三维建模和精细分类的效果和性能。



**Robotic 3D Scan Repository**

链接：http://kos.informatik.uni-osnabrueck.de/3Dscans/

这个数据集比较适合做SLAM研究，包含了大量的Riegl和Velodyne雷达数据

**佐治亚理工大型几何模型数据集**

链接：https://www.cc.gatech.edu/projects/large_models/

**PASCAL3D+**

链接：http://cvgl.stanford.edu/projects/pascal3d.html

包含了12中刚体分类，每一类超过了3000个实例。并且包含了对应的imageNet中每一类的图像。

**其他总结**

链接：https://github.com/timzhang642/3D-Machine-Learning

 

### 4.2. Competitions


#### [All Challenges](https://grand-challenge.org/challenges/)

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



---


## 5. Papers

### 5.1. Overview & Survey Papers

#### Medical Image

[1]    A. Sotiras, et.al., [“Deformable medical image registration: A survey,”]( https://ieeexplore.ieee.org/document/6522524 ) 2013.

[2]    N. J. Tustison, et.al., [“Learning image-based spatial transformations via convolutional neural networks : A review,” ]( https://www.sciencedirect.com/science/article/abs/pii/S0730725X19300037 )2019.

[3]    G. Haskins,et.al. [“Deep Learning in Medical Image Registration: A Survey,” ]( https://arxiv.org/pdf/1903.02026.pdf )2019.

[4]    M. Blendowski, et.al., [“Combining MRF-based deformable registration and deep binary 3D-CNN descriptors for large lung motion estimation in COPD patients,” ]( https://link.springer.com/article/10.1007/s11548-018-1888-2 )2019.



### 5.2. Key Algorithms

### 5.3. 




## 6. Key Conferences/Workshops/Journals

### 6.1. Conferences & Workshops

[**CVPR**](http://cvpr2020.thecvf.com/): IEEE International Conference on Computer Vision and Pattern Recognition

[**ICCV**](http://iccv2019.thecvf.com/): IEEE International Conference on Computer Vision

[**ECCV**](https://eccv2020.eu/): European Conference on Computer Vision

[**NeurIPS**]( https://nips.cc/): Conference on Neural Information Processing Systems

[**AAAI**]( http://www.aaai.org/ ): Association for the Advancement of Artificial Intelligence

[**ICML**]( https://icml.cc/): International Conference on Machine Learning

[**ICPR**]( https://www.icpr2020.it/): International Conference on Pattern Recognition

[**IJCNN**]( https://www.ijcnn.org/): International Joint Conference on Neural Networks

[**ICIP**](http://2019.ieeeicip.org/):  IEEE International Conference on Image Processing 

[**IJCAI**](https://www.ijcnn.org/): International Joint Conferences on Artificial Intelligence 

[**ICRA**](https://www.icra2020.org/): IEEE International Conference on Robotics and Automation

[**International Conference on 3D Vision**](http://3dv19.gel.ulaval.ca/)

[**WACV**](https://wacv20.wacv.net/): Winter Conference on Applications of Computer Vision



#### Biomedical image

[**MICCAI**]( http://www.miccai.org/ ): International Conference on Medical Image Computing and Computer Assisted Intervention

[**IPMI**](http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=77376): Information Processing in Medical Imaging

[**ISBI**]( http://2020.biomedicalimaging.org/): International Symposium on Biomedical Imaging 

[**Medical Imaging SPIE**](https://spie.org/conferences-and-exhibitions/medical-imaging?SSO=1 )



#### Remote Sensing

[ISPRS-2020](http://www.isprs2020-nice.com/)

#### Point Cloud(！该部分需要重新编辑)

点云配准主要应用于工业制造业的逆向工程、古文物修复、医学三维图像构建等领域。研究内容是属于计算机视觉领域的研究范畴。国际上的会议如计算机视觉三大顶会ICCV、CVPR、ECCV等都会有相关技术，除此之外，还有ACCV、BMVC、SSVM等认可度也比较高。

### 6.2. Journals

[IEEE Transactions on Pattern Analysis and Machine Intelligence](https://www.computer.org/csdl/journal/tp)

[International Journal of Computer Vision](https://link.springer.com/journal/11263)

[ISPRS Journal of Photogrammetry and Remote Sensing](https://www.journals.elsevier.com/isprs-journal-of-photogrammetry-and-remote-sensing)

#### Biomedical image

[**TMI**](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=42): IEEE Transactions on Medical Imaging

[**MIA**](https://www.journals.elsevier.com/medical-image-analysis/): Medical Image Analysis

[**TIP**](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83): IEEE Transactions on Image Processing

[**TBME**](https://tbme.embs.org/): IEEE Transactions on Biomedical Engineering

[**BOE**](https://www.osapublishing.org/boe/home.cfm): Biomedical Optics Express

[**JHBHI**](https://jbhi.embs.org/): Journal of Biomedical and Health Informatics



#### Remote Sensing

[Remote Sensing of Environment](https://www.journals.elsevier.com/remote-sensing-of-environment)

[ISPRS Journal of Photogrammetry And Remote Sensing](https://www.journals.elsevier.com/isprs-journal-of-photogrammetry-and-remote-sensing/)

[IEEE Transactions on Geoscience And Remote Sensing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=36)

[International Journal of Applied Earth Observation and Geoinformation](https://www.journals.elsevier.com/international-journal-of-applied-earth-observation-and-geoinformation/)

[IEEE Geoscience and Remote Sensing Letters](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8859)

[IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=4648334)

[Remote sensing](https://www.mdpi.com/journal/remotesensing)

[GIScience & Remote Sensing](http://www.bellpub.com/msrs/)

[Photogrammetric engineering and remote sensing](https://www.sciencedirect.com/journal/photogrammetric-engineering-and-remote-sensing)

[International journal of remote sensing](https://www.researchgate.net/journal/0143-1161_International_Journal_of_Remote_Sensing)

[Remote Sensing Letters](https://www.scimagojr.com/journalsearch.php?q=19700201680&tip=sid)

[Journal of Applied Remote Sensing](https://jars.msubmit.net/cgi-bin/main.plex)



#### **Point Cloud(！该部分需要重新编辑)**

IEEE旗下的TPAMI，TIP等，还有SIAM Journal Image Sciences，Springer那边有IJCV

