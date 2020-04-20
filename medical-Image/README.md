Mediacl image registration related books, papers, videos, and toolboxes. 

[![Stars](https://img.shields.io/github/stars/youngfish42/image-registration-resources.svg?color=orange)](https://github.com/youngfish42/image-registration-resources/stargazers) 
[![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86%E6%8C%87%E5%8C%97-blue)](https://zhuanlan.zhihu.com/Image-Registration) 
[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
[![License](https://img.shields.io/github/license/youngfish42/image-registration-resources.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) 

[**Image registration**](https://en.wikipedia.org/wiki/Image_registration) is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, and from different sensors, times, depths, or viewpoints.

It is used in computer vision, medical imaging, military automatic target recognition, compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from different measurements. 

Many thanks to **[Yochengliu](https://github.com/Yochengliu)**  [awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis) ,  **[hoya012](https://github.com/hoya012)**  [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)     and  **[Amusi](https://github.com/amusi)** [awesome-object-detection](https://github.com/amusi/awesome-object-detection)

# **Content**
1. [Paper Lists](#paper-lists)
2. [Datasets & Competitions](#datasets--competitions)
3. [Conferences/Workshops/Journals](#conferencesworkshopsjournals)
4. [Tools](#tools)
5. [Books & Tutorials](#books--tutorials)
6. [Courses](#courses)
7. [How to contact us](#how-to-contact-us)

# **Paper Lists**

A paper list of Medical Image registration. 

##  Keywords 

 **`medi.`**: medical image |  **`nat.`**: natural image |  **`rs.`**: remote sensing   |  **`pc.`**: point cloud

 **`data.`**: dataset  |   **`dep.`**: deep learning

 **`oth.`**: other, including  correspondence, mapping, matching, alignment...

Statistics: 🔥 code is available & stars >= 100  |  ⭐ citation >= 50

## Overview & Survey Papers

[[CVPR](https://arxiv.org/abs/1811.11397)] DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds. [[code](https://ai4ce.github.io/DeepMapping/)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. [[pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**]

## 2020

[[CVPR](https://arxiv.org/abs/2001.05119)] Learning multiview 3D point cloud registration. [[code](https://github.com/zgojcic/3D_multiview_reg)] [**`pc.`**]

## 2019

[[CVPR](https://arxiv.org/abs/1811.11397)] DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds. [[code](https://ai4ce.github.io/DeepMapping/)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. [[pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**]

# **Datasets & Competitions**
## Datasets
|                           Dataset                            | Number | Modality  |     Region     |     Format      |
| :----------------------------------------------------------: | :----: | :-------: | :------------: | :-------------: |
|             [DIRLAB](https://www.dir-lab.com/ )              |   10   |  4D  CT   |      Lung      |      .img       |
|                          [LPBA40]()                          |   40   |  3D  MRI  |    T1 Brain    | .img+.hdr  .nii |
|        [IBSR18](https://www.nitrc.org/projects/ibsr/)        |   18   |  3D  MRI  |    T1 Brain    |    .img+.hdr    |
|             [EMPIRE](http://empire10.isi.uu.nl/)             |   30   |   4D CT   |      Lung      |    .mhd+.raw    |
|                          [LiTS]( )                           |  131   |   3D CT   |     Liver      |      .nii       |
| [CT-scans-before-and-after](https://www.kaggle.com/kmader/ct-scans-before-and-after) |        |           |                |                 |
|            [Openi](https://openi.nlm.nih.gov/faq)            |        |   X-ray   |                |                 |
| [POPI](https://www.creatis.insa-lyon.fr/rio/popi-model?action=show&redirect=popi) |   6    |   4D CT   |                |                 |
|            [NLST](https://cdas.cancer.gov/nlst/)             |        |    CT     |      Lung      |                 |
|  [ADNI](http://adni.loni.usc.edu/data-samples/access-data/)  |        |  3D MRI   |     Brain      |                 |
|            [OASIS](http://www.oasis-brains.org/)             |        |  3D MRI   |     Brain      |                 |
| [ABIDE](http://preprocessed-connectomes-project.org/abide/)  |        |  3D MRI   |     Brain      |                 |
| [ADHD200](http://neurobureau.projects.nitrc.org/ADHD200/Introduction.html) |        |           |                |                 |
|    [CUMC12](https://www.synapse.org/#!Synapse:syn3207203)    |   12   |  3D MRI   |     Brain      |    .img+.hdr    |
|    [MGH10](https://www.synapse.org/#!Synapse:syn3207203)     |   10   |  3D MRI   |     Brain      |    .img+.hdr    |
|         [FIRE](https://www.ics.forth.gr/cvrl/fire/)          |  134   | 2D fundus |     Retina     |      .jpg       |
| [MSD](https://drive.google.com/open?id=17IiuM74HPj1fsWwkAfq-5Rc6r5vpxUJF) |        |    CT     |     Liver      |                 |
| [BFH](https://drive.google.com/open?id=17IiuM74HPj1fsWwkAfq-5Rc6r5vpxUJF) |   92   |    CT     |     Liver      |                 |
| [SLIVER](https://drive.google.com/open?id=1xQMmYk9S8En2k_uavytuHeeSmN253jKo) |   20   |    CT     |     Liver      |                 |
| [LSPIG](https://drive.google.com/open?id=1xQMmYk9S8En2k_uavytuHeeSmN253jKo) |   17   |    CT     |     Liver      |                 |
|         [OAI](http://oai.epi-ucsf.org/datarelease/)          | 20000+ |  3D MRI   | Osteoarthritis |                 |

## Competitions


### [All Challenges](https://grand-challenge.org/challenges/)

#### 2019 

[CuRIOUS:2019](https://curious2019.grand-challenge.org/) | [Official solution](https://arxiv.org/ftp/arxiv/papers/1904/1904.10535.pdf)

> 1 Register pre-operative MRI to iUS before tumor resection  
> 2 Register iUS after tumor resection to iUS before tumor resection  

[ANHIR:2019](https://anhir.grand-challenge.org/) | [Official solution](https://www.researchgate.net/publication/332428245_Automatic_Non-rigid_Histological_Image_Registration_challenge)

> IEEE International Symposium on Biomedical Imaging (ISBI) 2019  
> High-resolution (up to 40x magnification) whole-slide images of tissues (lesions, lung-lobes, mammary-glands) were acquired - the original size of our images is up to 100k x 200k pixels. The acquired images are organized in sets of consecutive sections where each slice was stained with a different dye and any two images within a set can be meaningfully registered.

#### 2018 

[iChallenges ](https://ichallenges.grand-challenge.org/) 

[Continuous Registration Challenge](https://continuousregistration.grand-challenge.org/) 

[Multi-shell Diffusion MRI Harmonisation Challenge 2018 (MUSHAC)](https://projects.iq.harvard.edu/cdmri2018/challenge)

#### 2010 

[EMPIRE10](http://empire10.isi.uu.nl/)

# **Conferences/Workshops/Journals**

## Journals
[//]:可以的话对会议进行一些介绍，比如涉及的领域，IF等。
- [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://www.computer.org/csdl/journal/tp)

- [International Journal of Computer Vision](https://link.springer.com/journal/11263)

## Conferences/Workshops

[//]:追踪最近的会议，罗列到这里

- [**CVPR**](http://cvpr2020.thecvf.com/): IEEE International Conference on Computer Vision and Pattern Recognition

- [**ICCV**](http://iccv2019.thecvf.com/): IEEE International Conference on Computer Vision

# **Tools**
## Open source libraries
[c++] [**ITK**](https://itk.org/): Segmentation & Registration Toolkit

An open-source, cross-platform system  that provides developers  with an extensive suite of software tools for image  analysis.  Developed through extreme  programming methodologies. ITK employs  leading-edge algorithms for registering  and segmenting multidimensional data.  

[c++] [Python] [Java] [**SimpleITK**](http://www.simpleitk.org/): a simplified layer built on top of ITK.

[c++] [**ANTs**](http://stnava.github.io/ANTs/): Advanced Normalization Tools.  

Image registration with variable transformations (elastic,  diffeomorphic, diffeomorphisms, unbiased) and similarity metrics  (landmarks, cross-correlation, mutual information, etc.). Image  segmentation with priors & nonparametric, multivariate models.   

[c++] [**Elastix**](http://elastix.isi.uu.nl/):  open source software, based on the well-known [ITK](http://www.itk.org) . 

The software consists of a collection of algorithms that are commonly used to solve (medical) image registration problems.  [**[manual]**](http://elastix.isi.uu.nl/download/elastix_manual_v4.8.pdf) 

[C++] [Python] [Java] [R] [Ruby] [Lua] [Tcl] [C#] [**SimpleElastix**](http://simpleelastix.github.io/): a medical image registration library that makes  state-of-the-art image registration really easy to do in languages like  Python, Java and R. 

[**3D slicer**](https://www.slicer.org/) :  an open source software platform for  medical image informatics, image processing, and three-dimensional  visualization. Built over two decades through support from the  National Institutes of Health and a worldwide developer community, Slicer brings free, powerful cross-platform processing tools to  physicians, researchers, and the general public.  

---

# **Books & Tutorials**    

## Books

Zhenhuan Zhou, et.al: [ **A software guide for medical image segmentation and registration algorithm. 医学图像分割与配准(ITK实现分册)**](https://vdisk.weibo.com/s/FQyto0RT-heb) 
Part Ⅱ introduces the most basic network and architecture of medical registration algorithms **(Chinese Version)**.

[2-D and 3-D Image Registration for Medical, Remote Sensing, and Industrial Applications](http://www.researchgate.net/profile/Rachakonda_Poojitha/post/How_to_reconstruct_a_3D_image_from_two_2D_images_of_the_same_scene_taken_from_the_same_camera/attachment/59d61d9d6cda7b8083a16a8f/AS%3A271832186327046%401441821251591/download/2-D+and+3-D+Image+Registration+for+Medical%2C+Remote+Sensing%2C+and+Industrial+Applications.pdf) by A. Ardeshir Goshtasby

[医学图像配准技术与应用](https://book.douban.com/subject/26411955/) by 吕晓琪    

[Intensity-based 2D-3D Medical Image Registration](https://blackwells.co.uk/bookshop/product/9783639119541) by Russakoff, Daniel

[Biomedical Image Registration](https://www.springer.com/gb/book/9783642143656) by Fischer, Dawant, Lorenz

[Medical Image Registration](https://wordery.com/medical-image-registration-joseph-v-hajnal-9780849300646) by  Hajnal, Joseph V.

[Deep Learning for Medical Image Analysis](https://www.elsevier.com/books/deep-learning-for-medical-image-analysis/zhou/978-0-12-810408-8) (part IV)


## Tutorials

- [**Medical Image Registration**](https://github.com/natandrade/Tutorial-Medical-Image-Registration) 

- [MICCAI2019] [**learn2reg**](https://github.com/learn2reg/tutorials2019) [PDF](https://github.com/learn2reg/tutorials2019/blob/master/slides)

- [kaggle:2016] [**Image registration, the R way, (almost) from scratch**](https://www.kaggle.com/vicensgaitan/image-registration-the-r-way)

- [kaggle:2018] [**X-Ray Patient Scan Registration**](https://www.kaggle.com/kmader/x-ray-patient-scan-registration)

- [MICCAI2019] [**Autograd Image Registration Laboratory**](https://github.com/airlab-unibas/MICCAITutorial2019)

- [MIT] [**HST.582J**](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-582j-biomedical-signal-and-image-processing-spring-2007/)  Biomedical Signal and Image Processing [PDF](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-582j-biomedical-signal-and-image-processing-spring-2007/lecture-notes/l16_reg1.pdf) 


## Blogs

### [图像配准指北](https://zhuanlan.zhihu.com/Image-Registration)

- [图像配准综述](https://zhuanlan.zhihu.com/p/80985475) 

- [基于深度学习的医学图像配准综述](https://zhuanlan.zhihu.com/p/70820773) 

- [基于深度学习和图像引导的医学图像配准](https://zhuanlan.zhihu.com/p/82423947) 

- [图像配准：从SIFT到深度学习](https://zhuanlan.zhihu.com/p/75784915) 

- [点云配准综述](https://zhuanlan.zhihu.com/p/91275450) 

- 图像配准会议介绍@ [MICCAI2019](https://zhuanlan.zhihu.com/p/87781312) / [CVPR2019](https://zhuanlan.zhihu.com/p/78798607) / [ICCV2019](https://zhuanlan.zhihu.com/p/80529725) / [NeurIPS2019](https://zhuanlan.zhihu.com/p/81658522)

- [Image Registration: From SIFT to Deep Learning]( https://blog.sicara.com/image-registration-sift-deep-learning-3c794d794b7a)

# **Courses**

- [16-822: Geometry-based Methods in Vision](http://www.cs.cmu.edu/~hebert/geom.html)

- [VALSE 2018](https://zhuanlan.zhihu.com/p/38611920): 2017以来的2D to 3D 工作

---

# **How to contact us**

We have QQ Group [【配准萌新交流群】](https://jq.qq.com/?_wv=1027&k=5r40AsF) （群号 869211738）

and Wechat Group 【配准交流群】（**已满员**） for comunications.



**More items will be added to the repository**.
Please feel free to suggest other key resources by opening an issue report,
submitting a pull request, or dropping me an email @ (im.young@foxmail.com).
Enjoy reading!