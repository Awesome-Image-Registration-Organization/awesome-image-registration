# Awesome Image Registration

A curated list of image registration related books, papers, videos, and toolboxes 

[![Stars](https://img.shields.io/github/stars/youngfish42/image-registration-resources.svg?color=orange)](https://github.com/youngfish42/image-registration-resources/stargazers)  [![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86%E6%8C%87%E5%8C%97-blue)](https://zhuanlan.zhihu.com/Image-Registration)  [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re) [![License](https://img.shields.io/github/license/youngfish42/image-registration-resources.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) 

[**Image registration**](https://en.wikipedia.org/wiki/Image_registration) is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, and from different sensors, times, depths, or viewpoints.

It is used in computer vision, medical imaging, military automatic target recognition, compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from different measurements. 

[toc]





[**Paper Lists**](#Paper-Lists) 

- [2024](#2024)
- [2023](#2023)
- [2022](#2022)
- [2021](#2021)
- [2020](#2020)
- [2019](#2019)
- [2018](#2018)
- [2017](#2017)
- [Before 2016](#before_2016)

**[Learning Resources](#Learning-Resources)**

- [Datasets](#Datasets)
- [Competitions](#Competitions)
- [Toolbox](#Toolbox)
- [Books](#Books)
- [Tutorials](#Tutorials)
- [Blogs](#Blogs)
- [Courses Seminars and Videos](#Courses-Seminars-and-Videos)
- [Conferences and Workshops](#Conferences-and-Workshops)
- [Journals](#Journals)



---

# Paper Lists

A paper list of image registration. 

###  Keywords 

 **`medi.`**: medical image |  **`nat.`**: natural image |  **`rs.`**: remote sensing   |  **`pc.`**: point cloud

 **`data.`**: dataset  |   **`dep.`**: deep learning

 **`oth.`**: other, including  correspondence, mapping, matching, alignment...

Statistics: :fire:  code is available & stars >= 100  |  :star: citation >= 50



### Update log

*Last updated: 2024/04/30*

*2024/04/30* - update recent papers on [TPAMI](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AIEEE_Trans._Pattern_Anal._Mach._Intell.%3A)/[MICCAI](https://dblp.org/search?q=registra%20venue%3AMICCAI%3A)/[CVPR](https://dblp.org/search?q=registra%20%20venue%3ACVPR%3A)/[ICCV](https://dblp.org/search?q=registra%20venue%3AICCV%3A)/[ECCV](https://dblp.org/search?q=registra%20venue%3AECCV%3A)/[AAAI](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3AAAAI%3A)/[NeurIPS](https://dblp.org/search?q=registra%20venue%3ANeurIPS%3A)/[MIA](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AMedical_Image_Anal.%3A)

*2023/03/02* - add papers according to [3D-PointCloud](https://github.com/zhulf0804/3D-PointCloud), update recent papers on CVPR/ECCV 2022

*2022/07/27* - update recent papers on ECCV 2022

*2022/07/14* - add the corresponding open source code for the 2022 papers

*2022/07/12* - update recent papers on AAAI 2022 and add information about competitions

*2022/06/19* - update recent TPAMI papers (2017-2021) about image registration according to dblp search engine, update recent MICCAI papers  about image registration according to  [MICCAI-OpenSourcePapers](https://github.com/JunMa11/MICCAI-OpenSourcePapers).

*2022/06/18* - update recent papers (2017-2021) on CVPR/ICCV/ECCV/AAAI/NeurIPS/MIA about image registration according to dblp search engine.

*2022/06/18* - update papers (2020-2022) about point cloud registration from [awesome-point-cloud-analysis-2023](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2023).

*2020/04/20* - update recent papers (2017-2020) about point cloud registration and make some diagram about history of image registration.



## 2024

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28601)] Test-Time Adaptation via Style and Structure Guidance for Histological Image Registration. 

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/27775)] PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations. [**`pc.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/27782)] SuperJunction: Learning-Based Junction Detection for Retinal Image Registration.

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28446)] SPEAL: Skeletal Prior Embedded Attention Learning for Cross-Source Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10380454)] RIGA: Rotation-Invariant and Globally-Aware Descriptors for Point Cloud Registration. [**`pc.`**]

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523003262)] Placental vessel segmentation and registration in fetoscopy: Literature review and MICCAI FetReg2021 challenge findings.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002876)] Automatic registration with continuous pose updates for marker-less surgical navigation in spine surgery. 

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002980)] Residual Aligner-based Network (RAN): Motion-separable structure for coarse-to-fine discontinuous deformable registration. [[Code](https://github.com/jianqingzheng/res_aligner_net)] [**`medi.`**]



## 2023

[[NeurIPS](http://papers.nips.cc/paper_files/paper/2023/hash/abf37695a4562ac4c05194d717d47eec-Abstract-Datasets_and_Benchmarks.html)] Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration.  [**`data.` ** **`pc.`**]

[[NeurIPS](https://proceedings.neurips.cc//paper_files/paper/2023/hash/43069caa6776eac8bca4bfd74d4a476d-Abstract-Conference.html)] SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation. [**`pc.`**]

[[NeurIPS](https://proceedings.neurips.cc//paper_files/paper/2023/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html)] Non-Rigid Shape Registration via Deep Functional Maps Prior.

[[NeurIPS](https://proceedings.neurips.cc//paper_files/paper/2023/hash/3a2d1bf9bc0a9794cf82c1341a7a75e6-Abstract-Conference.html)] E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning. [**`pc.`**]

[[NeurIPS](https://proceedings.neurips.cc//paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html)] Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10377293)] DReg-NeRF: Deep Registration for Neural Radiance Fields.

[[ICCV](https://ieeexplore.ieee.org/document/10377094)] Rethinking Point Cloud Registration as Masking and Reconstruction. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10377707)] SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10378635)] AutoSynth: Learning to Generate 3D Training Data for Object Point Cloud Registration. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10377041)] Preserving Tumor Volumes for Unsupervised Medical Image Registration.

[[ICCV](https://ieeexplore.ieee.org/document/10377676)] Towards Saner Deep Image Registration.

[[ICCV](https://ieeexplore.ieee.org/document/10378313)] Point-TTA: Test-Time Adaptation for Point Cloud Registration Using Multitask Meta-Auxiliary Learning. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10378523/)] Chasing clouds: Differentiable volumetric rasterisation of point clouds as a  highly efficient and accurate loss for large-scale deformable 3D registration. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10378523/)] Center-Based Decoupled Point Cloud Registration for 6D Object Pose Estimation. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10378633)] 2D3D-MATR: 2D-3D Matching Transformer for Detection-free Registration between Images and Point Clouds. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10376887)] RegFormer: An Efficient Projection-Aware Transformer Network for Large-Scale Point Cloud Registration.  [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10377308)] Density-invariant Features for Distant Point Cloud Registration. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/10376539)] Batch-based Model Registration for Fast 3D Sherd Reconstruction.

[[ICCV](https://ieeexplore.ieee.org/document/10378216)] PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration. [**`pc.`**]

[[ICCV workshop](https://ieeexplore.ieee.org/document/10350601)] Occluded Gait Recognition via Silhouette Registration Guided by Automated Occlusion Degree Estimation.

[[CVPR](https://ieeexplore.ieee.org/document/10205493/)] BUFFER: Balancing Accuracy, Efficiency, and Generalizability in Point Cloud Registration. [**`pc.`**]

[[CVPR](https://ieeexplore.ieee.org/document/10203801)] Local-to-Global Registration for Bundle-Adjusting Neural Radiance Fields.

[[CVPR](https://ieeexplore.ieee.org/document/10203222)] ObjectMatch: Robust Registration using Canonical Object Correspondences.

[[CVPR](https://ieeexplore.ieee.org/document/10203464)] PEAL: Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration. [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2303.00369.pdf)] Indescribable Multi-modal Spatial Evaluator [[Code](https://github.com/Kid-Liet/IMSE)] [**`medi.`**]

[[CVPR](https://arxiv.org/pdf/2304.01514.pdf)] Robust Outlier Rejection for 3D Registration with Variational Bayes [[Code](https://github.com/Jiang-HB/VBReg)] [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2305.10854v1.pdf)] 3D Registration with Maximal Cliques [[Code](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)] [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2303.09950.pdf)] Deep Graph-based Spatial Consistency for Robust Non-rigid Point Cloud Registration [[Code](https://github.com/qinzheng93/GraphSCNet)] [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2303.13290v1.pdf)] Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2304.00467v1.pdf)] Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting [[Code](https://github.com/WHU-USI3DV/SGHR)] [**`pc.`**]

[[AAAI](https://arxiv.org/pdf/2301.00149v1.pdf)] Rethinking Rotation Invariance with Point Cloud Registration  [[Code](https://github.com/Crane-YU/rethink_rotation)] [**`pc.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25182)] Fourier-Net: Fast Image Registration with Band-Limited Deformation.

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25220)] Stroke Extraction of Chinese Character Based on Deep Structure Deformable Image Registration.

[[TPAMI](https://ieeexplore.ieee.org/abstract/document/10044259)] RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations [[Code](https://github.com/HpWang-whu/RoReg)] [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10256027)] DPCN++: Differentiable Phase Correlation Network for Versatile Pose Registration.

[[TPAMI](https://ieeexplore.ieee.org/document/10115040)] SC${2}$2-PCR++: Rethinking the Generation and Selection for Efficient and Robust Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/9878213)] Robust Point Cloud Registration Framework Based on Deep Graph Matching.  [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/9749887)] Multiway Non-Rigid Point Cloud Registration via Learned Functional Map Synchronization. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10091912)] QGORE: Quadratic-Time Guaranteed Outlier Removal for Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10097640)] Sparse-to-Dense Matching Network for Large-Scale LiDAR Point Cloud Registration [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10148795)] HRegNet: A Hierarchical Network for Efficient and Accurate Outdoor LiDAR Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/9965747)] $\mathcal {X}$-Metric: An N-Dimensional Information-Theoretic Framework for Groupwise Registration and Deep Combined Computing.

[[TPAMI](https://ieeexplore.ieee.org/document/9775606)] Learning General and Distinctive 3D Local Deep Descriptors for Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10076895)] GeoTransformer: Fast and Robust Point Cloud Registration With Geometric Transformer. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/9931659)] Cycle Registration in Persistent Homology With Applications in Topological Bootstrap.

[[TPAMI](https://ieeexplore.ieee.org/document/9705149)] STORM: Structure-Based Overlap Matching for Partial Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10145843)] MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion.

[[TPAMI](https://ieeexplore.ieee.org/document/9969937)] A New Outlier Removal Strategy Based on Reliability of Correspondence Graph for Fast Point Cloud Registration. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10246849)] Hunter: Exploring High-Order Consistency for Point Cloud Registration With Severe Outliers. [**`pc.`**]

[[TPAMI](https://ieeexplore.ieee.org/document/10049724)] Fast and Robust Non-Rigid Registration Using Accelerated Majorization-Minimization.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_63)] A Denoised Mean Teacher for Domain Adaptive Point Cloud Registration. [code](https://github.com/uncbiag/robot)  [**`pc.`**]

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_64)] Unsupervised 3D Registration Through Optimization-Guided Cyclical Self-training.  [code](https://github.com/multimodallearning/reg-cyclical-self-train)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_61)] Implicit Neural Representations for Joint Decomposition and Registration of Gene Expression Images in the Marmoset Brain. [code](https://gene-atlas.brainminds.jp/)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_68)] An Unsupervised Multispectral Image Registration Network for Skin Diseases [code](https://github.com/SH-Diao123/MSIR)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_58)] GSMorph: Gradient Surgery for Cine-MRI Cardiac Deformable Registration. [code](https://github.com/wulalago/GSMorph)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_65)] Inverse Consistency by Construction for Multistep Deep Registration. [code](https://github.com/uncbiag/ByConstructionICON)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_22)] Learning Expected Appearances for Intraoperative Registration During Neurosurgery. [code](https://github.com/rouge1616/ExApp/)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_73)] StructuRegNet: Structure-Guided Multimodal 2D-3D Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_53)] SAMConvex: Fast Discrete Optimization for CT Registration Using Self-supervised Anatomical Embedding and Correlation Pyramid

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_51)] Co-learning Semantic-Aware Unsupervised Segmentation for Pathological Image Registration. [code](https://github.com/brain-intelligence-lab/GIRNet)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57)] PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer. [code](https://github.com/Torbjorn1997/PIViT)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_69)] CortexMorph: Fast Cortical Thickness Estimation via Diffeomorphic Registration Using VoxelMorph [code](https://github.com/SCAN-NRAD/CortexMorph)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_71)] Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_62)] FSDiffReg: Feature-Wise and Score-Wise Diffusion-Guided Unsupervised Deformable Image Registration for Cardiac Images.  [code](https://github.com/xmed-lab/FSDiffReg.git)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_75)] WarpEM: Dynamic Time Warping for Accurate Catheter Registration in EM-Guided Procedures.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_33)] Regularized Kelvinlet Functions to Model Linear Elasticity for Image-to-Physical Registration of the Breast. 

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_72)] DISA: DIfferentiable Similarity Approximation for Universal Multimodal Registration.  [code](https://github.com/ImFusionGmbH/DISA-universal-multimodal-registration)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_66)] FocalErrorNet: Uncertainty-Aware Focal Modulation Network for Inter-modal Registration Error Estimation in Ultrasound-Guided Neurosurgery.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74)] X-Ray to CT Rigid Registration Using Scene Coordinate Regression. [code](https://github.com/Pragyanstha/SCR-Registration)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_60)] Nonuniformly Spaced Control Points Based on Variational Cardiac Image Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_59)] Progressively Coupling Network for Brain MRI Registration in Few-Shot Situation. 

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_70)] ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer. [code](https://github.com/ZAX130/SmileCode)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_55)] Importance Weighted Variational Cardiac MRI Registration Using Transformer and Implicit Prior

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_40)] TractCloud: Registration-Free Tractography Parcellation with a Novel Local-Global Streamline Point Cloud Representation

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_12)] A Novel Video-CTU Registration Method with Structural Point Similarity for FURS Navigation.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_49)] A Patient-Specific Self-supervised Model for Automatic X-Ray/CT Registration [code](https://github.com/BaochangZhang/PSSS_registration)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_74)] SPR-Net: Structural Points Based Registration for Coronary Arteries Across Systolic and Diastolic Phases

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000658)] Strain estimation in aortic roots from 4D echocardiographic images using medial modeling and deformable registration.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002165)] Colonoscopy 3D video dataset with paired depth from 2D-3D registration.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000014)] AMNet: Adaptive multi-level network for deformable registration of 3D brain MR images.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001007)] DuSFE: Dual-Channel Squeeze-Fusion-Excitation co-attention for cross-modality registration of cardiac SPECT and CT.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000907)] Semantic similarity metrics for image registration.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001779)] R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000725)] Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001950)] Prototypical few-shot segmentation for cross-institution male pelvic structures with spatial registration.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001858)] WarpPINN: Cine-MR image registration with physics-informed neural networks.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002220)] A robust and interpretable deep learning framework for multi-modal registration via keypoints.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002384)] PC-Reg: A pyramidal prediction-correction approach for large deformation image registration.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841522003061)] DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000476)] SpineRegNet: Spine Registration Network for volumetric MR and CT image by the joint estimation of an affine-elastic deformation field.

[[MIA](https://linkinghub.elsevier.com/retrieve/pii/S1361841522003206)] QACL: Quartet attention aware closed-loop learning for abdominal MR-to-CT synthesis via simultaneous registration.


## 2022

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19833-5_26)] Bayesian Tracking of Video Graphs Using Joint Kalman Smoothing and Registration.

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_2)] Learning-Based Point Cloud Registration for 6D Object Pose Estimation in the Real World. [**`pc.`**]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_20)] DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model. [[Code](https://github.com/diffusemorph/diffusemorph)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_26)] PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry. [[Code](https://github.com/Gardlin/PCR-CG)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_4)] Unsupervised Deep Multi-shape Matching. [[Code](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_2)] ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer. [[Code](https://github.com/apple/ml-aspanformer)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_6)] CMT: Context-Matching-Guided Transformer for 3D Tracking in Point Clouds. [[code](https://github.com/jasongzy/CMT)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_37)] A Comparative Study of Graph Matching Algorithms in Computer Vision.

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_22)] Self-supervised Learning of Visual Graph Matching.

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_8)] 3DG-STFM: 3D Geometric Guided Student-Teacher Feature Matching. [[Code](https://github.com/Ryan-prime/3DG-STFM)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_24)] Is Geometry Enough for Matching in Visual Localization?

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19781-9_2)] Explaining Deepfake Detection by Analysing Image Matching  [[Code](https://github.com/megvii-research/fst-matching)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_35)] Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching [[Code](https://github.com/ruc-aimc-lab/superretina)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_1)] DFNet: Enhance Absolute Pose Regression with Direct Feature Matching [[Code](https://github.com/activevisionlab/dfnet)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_41)] Unitail: Detecting, Reading, and Matching in Retail Scene

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_20)] Implicit field supervision for robust non-rigid shape matching [[Code](https://github.com/Sentient07/IFMatch)]

[[EECV](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_18)] Registration based Few-Shot Anomaly Detection [[Code](https://github.com/mediabrain-sjtu/regad)]

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_11)] Improving RGB-D Point Cloud Registration by Learning Multi-scale Local Linear Transformation [[Code](https://github.com/514DNA/LLT)] [**`pc.`**]

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_35)] PointCLM: A Contrastive Learning-based Framework for Multi-instance Point Cloud Registration [[Code](https://github.com/phdymz/PointCLM)] [**`pc.`**]

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_16)] SuperLine3D: Self-supervised Line Segmentation and Description for LiDAR Point Cloud [[Code](https://github.com/zxrzju/SuperLine3D)] [**`pc.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20086/19845)] Stochastic Planner-Actor-Critic for Unsupervised Deformable Image Registration. [[code](https://github.com/Algolzw/SPAC-Deformable-Registration)] [**`medi.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/19917)] DeTarNet: Decoupling Translation and Rotation by Siamese Network for Point Cloud Registration. [[code](https://github.com/ZhiChen902/DetarNet)] [**`pc.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/19951)] Deep Confidence Guided Distance for 3D Partial Shape Registration.

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20117)] Reliable Inlier Evaluation for Unsupervised Point Cloud Registration. [[code](https://github.com/supersyq/rienet)] [**`pc.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20189)] FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration. [[code](https://github.com/MegEngine/FINet)] [**`pc.`**]

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20250)] End-to-End Learning the Partial Permutation Matrix for Robust 3D Point Cloud Registration. [**`pc.`**]

[[CVPR](https://ieeexplore.ieee.org/document/9878458/)] Deterministic Point Cloud Registration via Novel Transformation Decomposition.  [**`pc.`**]

[[CVPR](https://ieeexplore.ieee.org/document/9879348/)] Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment. [[Code](https://github.com/uncbiag/Aladdin)]

[[CVPR](https://ieeexplore.ieee.org/document/9879560/)] Coherent Point Drift Revisited for Non-rigid Shape Matching and Registration. [[Code](https://github.com/AoxiangFan/GeneralizedCoherentPointDrift)]

[[CVPR](https://ieeexplore.ieee.org/document/9879941)] A variational Bayesian method for similarity learning in non-rigid image registration. [[Code](https://github.com/dgrzech/learnsim)]

[[CVPR](https://ieeexplore.ieee.org/document/9879546/)] Affine Medical Image Registration with Coarse-to-Fine Vision Transformer. [[Code](https://github.com/cwmok/C2FViT)]

[[CVPR](https://ieeexplore.ieee.org/document/9880332/)] Topology-Preserving Shape Reconstruction and Registration via Neural Diffeomorphic Flow. [[Code](https://github.com/Siwensun/Neural_Diffeomorphic_Flow--NDF)]

[[CVPR](https://ieeexplore.ieee.org/document/9878484/)] Global-Aware Registration of Less-Overlap RGB-D Scans. [[Code](https://github.com/2120171054/Global-Aware-Registration-of-Less-Overlap-RGB-D-Scans)]

[[CVPR](https://ieeexplore.ieee.org/document/9880256/)] Multi-instance Point Cloud Registration by Efficient Correspondence Clustering. [[Code](https://github.com/Gilgamesh666666/Multi-instance-Point-Cloud-Registration-by-Efficient-Correspondence-Clustering)] [**`pc.`**]

[[CVPR](https://ieeexplore.ieee.org/document/9879617)] NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration. [[Code](https://github.com/yifannnwu/NODEO-DIR)]

[[CVPR](https://ieeexplore.ieee.org/document/9878923)] RFNet: Unsupervised Network for Mutually Reinforcing Multi-modal Image Registration and Fusion.

[[CVPR](https://arxiv.org/pdf/2203.14517v1.pdf)] REGTR: End-to-end Point Cloud Correspondences with Transformers. [[code](https://github.com/yewzijian/RegTR)] [**`pc.`**]

[[CVPR](https://ieeexplore.ieee.org/document/9878510/)] SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration. [[code](https://github.com/ZhiChen902/SC2-PCR)] [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2202.06688.pdf)] Geometric Transformer for Fast and Robust Point Cloud Registration. [[code](https://github.com/qinzheng93/GeoTransformer)] [**`pc.`**] :fire:

[[CVPR](https://ieeexplore.ieee.org/document/9878922)] Lepard: Learning partial point cloud matching in rigid and deformable scenes [[Code](https://github.com/rabbityl/lepard)]

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_27)] Adapting the Mean Teacher for Keypoint-Based Lung Registration Under Geometric Domain Shifts.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_58)] Vol2Flow: Segment 3D Volumes Using a Sequence of Registration Flows.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_14)] Deformer: Towards Displacement Field Learning for Unsupervised Medical Image Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_5)] Dual-Branch Squeeze-Fusion-Excitation Module for Cross-Modality Registration of Cardiac SPECT and CT.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_7)] ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_29)] Learning Iterative Optimisation for Deformable Image Registration of Lung CT with Recurrent Convolutional Networks. 

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_22)] Weakly-Supervised Biomechanically-Constrained CT/MRI Registration of the Spine.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_15)] End-to-End Multi-Slice-to-Volume Concurrent Registration and Multimodal Generation

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_4)] On the Dataset Quality Control for Image Registration Evaluation.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_10)] DSR: Direct Simultaneous Registration for Multiple 3D Images

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_26)] Global Multi-modal 2D/3D Registration via Local Descriptors Learning.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_9)] Non-iterative Coarse-to-Fine Registration Based on Single-Pass Deep Cumulative Learning. [code](https://github.com/MungoMeng/Registration-NICE-Trans)

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_8)] An Optimal Control Problem for Elastic Registration and Force Estimation in Augmented Surgery. 

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_3)] Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-recurrence Brain Tumor MRI Scans.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_6)] Embedding Gradient-Based Optimization in Image Registration Networks.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_23)] Collaborative Quantization Embeddings for Intra-subject Prostate MR Image Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_21)] XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_11)] Multi-modal Retinal Image Registration Using a Keypoint-Based Vessel Structure Aligning Network.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_12)] A Deep-Discrete Learning Framework for Spherical Surface Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_13)] Privacy Preserving Image Registration.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_20)] LiftReg: Limited Angle 2D/3D Deformable Registration. 

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_30)] Electron Microscope Image Registration Using Laplacian Sharpening Transformer U-Net.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_2)] Double-Uncertainty Guided Spatial and Temporal Consistency Regularization Weighting for Learning-Based Abdominal Registration. 

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1)] SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_17)] Learning-Based US-MR Liver Image Registration with Spatial Priors

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_8)] Swin-VoxelMorph: A Symmetric Unsupervised Learning Model for Deformable Medical Image Registration Using Swin Transformer.

[[MIA](https://doi.org/10.1016/j.media.2021.102265)] Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas. [**`medi.`**] [[code](https://github.com/acasamitjana/3dhirest)]

[[MIA](https://doi.org/10.1016/j.media.2021.102292)] Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2022.102379)] Dual-stream pyramid registration network. [**`medi.`**] [[Unofficial code](https://github.com/olddriverjinx/reimplemention-of-dual-prnet)]

[[MIA](https://doi.org/10.1016/j.media.2022.102383)] Atlas-ISTN: Joint segmentation, registration and atlas construction with image-and-spatial transformer networks. [**`medi.`**]

[[NeurIPS](https://arxiv.org/pdf/2205.12796.pdf)] Non-rigid Point Cloud Registration with Neural Deformation Pyramid  [[Code](https://github.com/rabbityl/DeformationPyramid)]

[[NeurIPS](https://proceedings.neurips.cc//paper_files/paper/2022/hash/2e163450c1ae3167832971e6da29f38d-Abstract-Conference.html)] One-Inlier is First: Towards Efficient Position Encoding for Point Cloud Registration.

[[ICLR](https://arxiv.org/pdf/2203.02227.pdf)] Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration]

[[TIP](https://arxiv.org/pdf/2302.01109.pdf)] GraphReg: Dynamical Point Cloud Registration with Geometry-aware Graph Signal Processing [[Code](https://github.com/zikai1/GraphReg)]

[[TPAMI](https://ieeexplore.ieee.org/document/9878213)] Robust Point Cloud Registration Framework Based on Deep Graph Matching [[Code](https://github.com/fukexue/RGM)]

[[TPAMI](https://arxiv.org/pdf/2209.13252.pdf)] RIGA: Rotation-Invariant and Globally-Aware Descriptors for Point Cloud Registration

[[SIGGRAPH](https://arxiv.org/pdf/2207.00826.pdf)] ImLoveNet: Misaligned Image-supported Registration Network for Low-overlap Point Cloud Pairs

[[ACM MM](https://arxiv.org/pdf/2109.00182.pdf)] You Only Hypothesize Once: Point Cloud Registration with Rotation-equivariant Descriptors [[Code](https://github.com/HpWang-whu/YOHO)]

[[TVCG](https://arxiv.org/pdf/2108.02740.pdf)] WSDesc: Weakly Supervised 3D Local Descriptor Learning for Point Cloud Registration [[Code](https://github.com/craigleili/WSDesc)]

[[RAL](https://arxiv.org/pdf/2212.12745.pdf)] GraffMatch: Global Matching of 3D Lines and Planes for Wide Baseline LiDAR Registration

## 2021

[[CVPR](https://arxiv.org/pdf/2011.13005.pdf)] PREDATOR: Registration of 3D Point Clouds with Low Overlap. [[code-pytorch](https://github.com/ShengyuH/OverlapPredator)] [**`pc.`**] :fire:

[[CVPR](https://github.com/QingyongHu/SpinNet)] SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration. [[code-pytorch](https://github.com/QingyongHu/SpinNet)] [**`pc.`**] :fire:

[[CVPR](https://arxiv.org/abs/2103.04256)] Robust Point Cloud Registration Framework Based on Deep Graph Matching. [[code](https://github.com/fukexue/RGM)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/2103.05465)] PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency. [**`pc.`**]

[[CVPR](https://arxiv.org/abs/2103.15231)] ReAgent: Point Cloud Registration using Imitation and Reinforcement Learning.[**`pc.`**]

[[CVPR](https://arxiv.org/abs/2104.03501)] DeepI2P: Image-to-Point Cloud Registration via Deep Classification. [[code](https://github.com/lijx10/DeepI2P)] [**`pc.`**] :fire:

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Banani_UnsupervisedRR_Unsupervised_Point_Cloud_Registration_via_Differentiable_Rendering_CVPR_2021_paper.pdf)] UnsupervisedR&R: Unsupervised Point Cloud Registration via Differentiable Rendering. [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2008.09527.pdf)] PointNetLK Revisited. [[code](https://github.com/Lilac-Lee/PointNetLK_Revisited)] [**`pc.`**]

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Ali_RPSRNet_End-to-End_Trainable_Rigid_Point_Set_Registration_Network_Using_Barnes-Hut_CVPR_2021_paper.html)] RPSRNet: End-to-End Trainable Rigid Point Set Registration Network Using Barnes-Hut 2D-Tree Representation

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Camera-Space_Hand_Mesh_Recovery_via_Semantic_Aggregation_and_Adaptive_2D-1D_CVPR_2021_paper.html)] Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_Recurrent_Multi-View_Alignment_Network_for_Unsupervised_Surface_Registration_CVPR_2021_paper.html)] Recurrent Multi-View Alignment Network for Unsupervised Surface Registration.

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Spatiotemporal_Registration_for_Event-Based_Visual_Odometry_CVPR_2021_paper.html)] Spatiotemporal Registration for Event-Based Visual Odometry.

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Safadi_Learning-Based_Image_Registration_With_Meta-Regularization_CVPR_2021_paper.html)] Learning-Based Image Registration With Meta-Regularization. 

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Locally_Aware_Piecewise_Transformation_Fields_for_3D_Human_Mesh_Registration_CVPR_2021_paper.html)] Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration.

[[ICCV](https://arxiv.org/abs/2107.11992)] HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration. [[code](https://ispc-group.github.io/hregnet)] [**`pc.`**]

[[ICCV oral](https://arxiv.org/abs/2108.03257)] (Just) A Spoonful of Refinements Helps the Registration Error Go Down. [**`pc.`**]

[[ICCV](https://arxiv.org/abs/2108.11682)] A Robust Loss for Point Cloud Registration. [**`pc.`**]

[[ICCV](https://arxiv.org/abs/2109.04310)] Deep Hough Voting for Robust Global Registration. [**`pc.`**]

[[ICCV](https://arxiv.org/abs/2109.06619)] Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration.[**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_LSG-CPD_Coherent_Point_Drift_With_Local_Surface_Geometry_for_Point_ICCV_2021_paper.pdf)] LSG-CPD: Coherent Point Drift with Local Surface Geometry for Point Cloud Registration.[[code](https://github.com/ChirikjianLab/LSG-CPD)] [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf)] OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration.[[code](https://github.com/megvii-research/OMNet)] [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_DeepPRO_Deep_Partial_Point_Cloud_Registration_of_Objects_ICCV_2021_paper.pdf)] DeepPRO: Deep Partial Point Cloud Registration of Objects. [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Feature_Interactive_Representation_for_Point_Cloud_Registration_ICCV_2021_paper.pdf)] Feature Interactive Representation for Point Cloud Registration.[[code](https://github.com/Ghostish/BAT)] [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Jubran_Provably_Approximated_Point_Cloud_Registration_ICCV_2021_paper.pdf)] Provably Approximated Point Cloud Registration. [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Distinctiveness_Oriented_Positional_Equilibrium_for_Point_Cloud_Registration_ICCV_2021_paper.pdf)] Distinctiveness oriented Positional Equilibrium for Point Cloud Registration. [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Cao_PCAM_Product_of_Cross-Attention_Matrices_for_Rigid_Registration_of_Point_ICCV_2021_paper.pdf)] PCAM: Product of Cross-Attention Matrices for Rigid Registration of Point Clouds. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/9711216)] Generative Adversarial Registration for Improved Conditional Deformable Templates.

[[ICCV](https://ieeexplore.ieee.org/document/9710911)] Deep Hough Voting for Robust Global Registration.

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/16116)] Low-Rank Registration Based Manifolds for Convection-Dominated PDEs.

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/18031)] TAILOR: Teaching with Active and Incremental Learning for Object Registration.

[[NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html)] Accurate Point Cloud Registration with Robust Optimal Transport.

[[NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/2d3d9d5373f378108cdbd30a3c52bd3e-Abstract.html)] Shape Registration in the Time of Transformers.

[[NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/c85b2ea9a678e74fdc8bafe5d0707c31-Abstract.html)] CoFiNet: Reliable Coarse-to-fine Correspondences for Robust PointCloud Registration.

[[Robotics and Autonomous Systems](https://www.sciencedirect.com/science/article/abs/pii/S0921889021000191?via%3Dihub)] A Benchmark for Point Clouds Registration Algorithms [[code](https://github.com/iralabdisco/point_clouds_registration_benchmark?utm_source=catalyzex.com)] [**`pc.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2020.2983935)] Supervision by Registration and Triangulation for Landmark Detection.

[[TPAMI](https://doi.org/10.1109/TPAMI.2020.3043769)] Acceleration of Non-Rigid Point Set Registration With Downsampling and Gaussian Process Regression.

[[TPAMI](https://doi.org/10.1109/TPAMI.2020.2978477)] Point Set Registration for 3D Range Scans Using Fuzzy Cluster-Based Metric and Efficient Global Optimization. 

[[TPAMI](https://doi.org/10.1109/TPAMI.2019.2940655)] Topology-Aware Non-Rigid Point Cloud Registration. 

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/013-Paper0479.html)] A Deep Discontinuity-Preserving Image Registration Network. [Code](https://github.com/cistib/DDIR) [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/013-Paper0479.html)] A Deep Network for Joint Registration and Parcellation of Cortical Surfaces.  [Code](https://github.com/zhaofenqiang/JointRegAndParc) [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/099-Paper0422.html)] Conditional Deformable Image Registration with Convolutional Neural Network. [[code](https://github.com/cwmok/Conditional_LapIRN)] [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/122-Paper0642.html)] Cross-modal Attention for MRI and Ultrasound Volume Registration. [[code](https://github.com/DIAL-RPI/Attention-Reg)] [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/178-Paper0585.html)] End-to-end Ultrasound Frame to Volume Registration. [[code](https://github.com/DIAL-RPI/FVR-Net)] [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/281-Paper0337.html)] Learning Unsupervised Parameter-specific Affine Transformation for Medical Images Registration. [[code](https://github.com/xuuuuuuchen/PASTA)] [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/337-Paper1007.html)] Multi-view analysis of unregistered medical images using cross-view transformers. [[code](https://github.com/gvtulder/cross-view-transformers)] [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/405-Paper2461.html)] Revisiting iterative highly efficient optimisation schemes in medical image registration. [[code](https://github.com/multimodallearning/iter_lbp)] [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/512-Paper0947.html)] Unsupervised Diffeomorphic Surface Registration and Non-Linear Modelling. [[code](https://gitlab.kuleuven.be/u0132345/deepdiffeomorphicfaceregistration)] [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.101983)] A hybrid, image-based and biomechanics-based registration approach to markerless intraoperative nodule localization during video-assisted thoracoscopic surgery. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102231)] Real-time multimodal image registration with partial intraoperative point-set data. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102157)] Leveraging unsupervised image registration for discovery of landmark shape descriptor. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101822)] Weakly-supervised learning of multi-modal features for regularised iterative descent in 3D image registration. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102228)] Shape registration with learned deformations for 3D shape reconstruction from sparse and incomplete point clouds. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101941)] Variational multi-task MRI reconstruction: Joint reconstruction, registration and super-resolution. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101815)] A novel approach to 2D/3D registration of X-ray images using Grangeat's relation. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101845)] Biomechanically constrained non-rigid MR-TRUS prostate registration using deep learning based 3D point cloud matching. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101917)] Fracture reduction planning and guidance in orthopaedic trauma surgery via multi-body image registration. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102139)] CNN-based lung CT registration with multiple anatomical constraints. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101878)] End-to-end multimodal image registration via reinforcement learning. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101817)] Difficulty-aware hierarchical convolutional neural networks for deformable registration of brain MR images. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102036)] CycleMorph: Cycle consistent unsupervised deformable image registration. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101930)] Rethinking medical image reconstruction via shape prior, going deeper and faster: Deep joint indirect registration and reconstruction. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102181)] Deformation analysis of surface and bronchial structures in intraoperative pneumothorax using deformable mesh registration. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101823)] Re-Identification and growth detection of pulmonary nodules without image registration using 3D siamese neural networks. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101939)] Image registration: Maximum likelihood, minimum entropy and deep learning. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101919)] ProsRegNet: A deep learning framework for registration of MRI and histopathology images of the prostate. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.101957)] 3D Registration of pre-surgical prostate MRI and histopathology images via super-resolution volume reconstruction. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101884)] A deep learning framework for pancreas segmentation with multi-atlas registration and 3D level-set. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102041)] Anatomy-guided multimodal registration by learning segmentation without ground truth: Application to intraprocedural CBCT/MR liver segmentation and registration. [**`medi.`**]



## 2020

[[CVPR](https://arxiv.org/abs/2001.05119)] Learning multiview 3D point cloud registration. [[code](https://github.com/zgojcic/3D_multiview_reg)] [**`pc.`**] :fire:

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf)] SampleNet: Differentiable Point Cloud Sampling. [[code](https://github.com/itailang/SampleNet)] [**`pc.`**] :fire:

[[CVPR](https://arxiv.org/abs/2005.01014)] Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences. [[code](https://github.com/XiaoshuiHuang/fmr)]  [**`pc.`**]

[[CVPR oral](https://arxiv.org/abs/2004.11540)] Deep Global Registration. [**`pc.`**]

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Arar_Unsupervised_Multi-Modal_Image_Registration_via_Geometry_Preserving_Image-to-Image_Translation_CVPR_2020_paper.html)] Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation.

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Eisenberger_Smooth_Shells_Multi-Scale_Shape_Registration_With_Functional_Maps_CVPR_2020_paper.html)] Smooth Shells: Multi-Scale Shape Registration With Functional Maps.

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Iglesias_Global_Optimality_for_Point_Set_Registration_Using_Semidefinite_Programming_CVPR_2020_paper.html)] Global Optimality for Point Set Registration Using Semidefinite Programming.

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Mok_Fast_Symmetric_Diffeomorphic_Image_Registration_with_Convolutional_Neural_Networks_CVPR_2020_paper.html)] Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks.

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Pais_3DRegNet_A_Deep_Neural_Network_for_3D_Point_Registration_CVPR_2020_paper.html)] 3DRegNet: A Deep Neural Network for 3D Point Registration.

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_DeepFLASH_An_Efficient_Network_for_Learning-Based_Medical_Image_Registration_CVPR_2020_paper.html)] DeepFLASH: An Efficient Network for Learning-Based Medical Image Registration.

[[CVPR](https://openaccess.thecvf.com/content_CVPR_2020/html/Yao_Quasi-Newton_Solver_for_Robust_Non-Rigid_Registration_CVPR_2020_paper.html)] Quasi-Newton Solver for Robust Non-Rigid Registration.

[[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/970af30e481057c48f87e101b61e6994-Abstract.html)] LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration.

[[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html)] CoMIR: Contrastive Multimodal Image Representation for Registration. 

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58523-5_45)] Deep Complementary Joint Model for Complex Scene Registration and Few-Shot Segmentation on Medical Images.

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_23)] Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration.

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_16)] JSSR: A Joint Synthesis, Segmentation, and Registration System for 3D Multi-modal Image Alignment of Large-Scale Pathological CT Scans.

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_17)] A Closest Point Proposal for MCMC-based Probabilistic Surface Registration.

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_43)] DeepGMR: Learning Latent Gaussian Mixture Models for Registration.

[[3DV](https://arxiv.org/abs/2011.02229)] Registration Loss Learning for Deep Probabilistic Point Set Registration. [[code-pytorch](https://github.com/felja633/RLLReg)] [**`pc.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2019.2908635)] Aggregated Wasserstein Distance and State Registration for Hidden Markov Models.

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_15)] MvMM-RegNet: A New Image Registration Framework Based on Multivariate Mixture Model and Neural Network Estimation [[code](https://zmiclab.github.io/projects.html)] [**`medi.`**]

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_19)] Highly Accurate and Memory Efficient Unsupervised Learning-Based Discrete CT Registration Using 2.5D Displacement Search [[code](https://github.com/multimodallearning/pdd2.5/)] [**`medi.`**]

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_32)] Generalizing Spatial Transformers to Projective Geometry with Applications to 2D/3D Registration [[code](https://github.com/gaocong13/Projective-Spatial-Transformers)] [**`medi.`**]

[[MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-59719-1_70)] Non-Rigid Volume to Surface Registration Using a Data-Driven Biomechanical Model [[code](https://gitlab.com/nct_tso_public/Volume2SurfaceCNN)] [**`medi.`**]


[[MIA](https://doi.org/10.1016/j.media.2019.101564)] Hubless keypoint-based 3D deformable groupwise registration. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101698)] Multi-atlas image registration of clinical data with automated quality assessment using ventricle segmentation. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101711)] Groupwise registration with global-local graph shrinkage in atlas construction. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2020.101763)] SLIR: Synthesis, localization, inpainting, and registration for image-guided thermal ablation of liver tumors. [**`medi.`**]



## 2019

[[CVPR](https://arxiv.org/abs/1811.11397)] DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds. [[code](https://ai4ce.github.io/DeepMapping/)] [**`pc.`**] :fire:

[[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. [[code-pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**] :fire:

[[CVPR](https://arxiv.org/abs/1904.03483)] SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. [[matlab](https://github.com/intellhave/SDRSAC)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1811.10136)] FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. [[code](https://bitbucket.org/gaowei19951004/poser/src/master/)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Robust & Efficient Point Cloud Registration using PointNet. [[code-pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**] :fire:

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] 3D Local Features for Direct Pairwise Registration. [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/html/Liao_Multiview_2D3D_Rigid_Registration_via_a_Point-Of-Interest_Network_for_Tracking_CVPR_2019_paper.html)] Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation.

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/html/Niethammer_Metric_Learning_for_Image_Registration_CVPR_2019_paper.html)] Metric Learning for Image Registration.

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/html/Shen_Networks_for_Joint_Affine_and_Non-Parametric_Image_Registration_CVPR_2019_paper.html)] Networks for Joint Affine and Non-Parametric Image Registration.

[[ICCV](https://arxiv.org/abs/1905.04153v2)] DeepVCP: An End-to-End Deep Neural Network for 3D Point Cloud Registration. [**`pc.`**]

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Golyanik_Accelerated_Gravitational_Point_Set_Alignment_With_Altered_Physical_Laws_ICCV_2019_paper.pdf)] Accelerated Gravitational Point Set Alignment with Altered Physical Laws. [**`pc.`**]

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)] Deep Closest Point: Learning Representations for Point Cloud Registration. [**`pc.`**]

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Prokudin_Efficient_Learning_on_Point_Clouds_With_Basis_Point_Sets_ICCV_2019_paper.pdf)] Efficient Learning on Point Clouds with Basis Point Sets. [[code](https://github.com/sergeyprokudin/bps)] [ **`pc.`**] :fire:

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Robust_Variational_Bayesian_Point_Set_Registration_ICCV_2019_paper.pdf)] Robust Variational Bayesian Point Set Registration. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/9008111)] Efficient and Robust Registration on the 3D Special Euclidean Group.

[[ICCV](https://ieeexplore.ieee.org/document/9010695)] Linearly Converging Quasi Branch and Bound Algorithms for Global Rigid Registration.

[[ICCV](https://ieeexplore.ieee.org/document/9008309)] A Deep Step Pattern Representation for Multimodal Retinal Image Registration.

[[ICCV](https://ieeexplore.ieee.org/document/9010680)] Recursive Cascaded Networks for Unsupervised Medical Image Registration.

[[ICCV](https://ieeexplore.ieee.org/document/9008291)] Automatic and Robust Skull Registration Based on Discrete Uniformization.

[[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/56f9f88906aebf4ad985aaec7fa01313-Abstract.html)] Arbicon-Net: Arbitrary Continuous Geometric Transformation Networks for Image Registration.

[[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/dd03de08bfdff4d8ab01117276564cc7-Abstract.html)] Recurrent Registration Neural Networks for Deformable Image Registration.

[[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/ebad33b3c9fa1d10327bb55f9e79e2f3-Abstract.html)] PRNet: Self-Supervised Learning for Partial-to-Partial Registration.

[[TPAMI](https://doi.org/10.1109/TPAMI.2018.2831670)] Efficient Registration of High-Resolution Feature Enhanced Point Clouds.

[[MICCAI](https://arxiv.org/abs/1907.10931)] Closing the Gap between Deep and Conventional Image Registration using Probabilistic Dense Displacement Networks [[code](https://github.com/multimodallearning/pdd_net)] [**`medi.`**]

[[ICRA](https://ieeexplore.ieee.org/abstract/document/8793857)] Robust low-overlap 3-D point cloud registration for outlier rejection. [[matlab](https://github.com/JStech/ICP)] [**`pc.`**]

[[ICRA](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_3.html)] Robust Generalized Point Set Registration Using Inhomogeneous Hybrid Mixture Models Via Expectation. [**`pc.`**]

[[ICRA](https://export.arxiv.org/abs/1810.01470)] CELLO-3D: Estimating the Covariance of ICP in the Real World. [**`pc.`**]




## 2018

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)] Density Adaptive Point Set Registration. [[code](https://github.com/felja633/DARE)] [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Vongkulbhisal_Inverse_Composition_Discriminative_CVPR_2018_paper.pdf)] Inverse Composition Discriminative Optimization for Point Cloud Registration. [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/html/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.html)] An Unsupervised Learning Model for Deformable Medical Image Registration.

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Supervision-by-Registration_An_Unsupervised_CVPR_2018_paper.html)] Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors.

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/html/Jiang_CNN_Driven_Sparse_CVPR_2018_paper.html)] CNN Driven Sparse Multi-Level B-Spline Image Registration.

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/html/Raposo_3D_Registration_of_CVPR_2018_paper.html)] 3D Registration of Curves and Surfaces Using Local Differential Information.

[[AAAI](https://web.archive.org/web/*/https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16085)] Dilated FCN for Multi-Agent 2D/3D Medical Image Registration.

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhou_Learning_and_Matching_ECCV_2018_paper.pdf)] Learning and Matching Multi-View Descriptors for Registration of Point Clouds. [**`pc.`**]

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. [[code-tensorflow](https://github.com/yewzijian/3DFeatNet)] [**`pc.`**] :fire:

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf)] Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search. [**`pc.`**]

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)] HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration. [**`pc.`**]

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-01216-8_4)] Robust Image Stitching with Multiple Registrations.


[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2773482)] Guaranteed Outlier Removal for Point Cloud Registration with Correspondences. [**`pc.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2730205)] Collocation for Diffeomorphic Deformations in Medical Image Registration. [**`medi.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2748125)] Hierarchical Sparse Representation for Robust Image Registration.

[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2654245)] Multiresolution Search of the Rigid Motion Space for Intensity-Based Registration. 

[[3DV](https://arxiv.org/abs/1808.00671)] PCN: Point Completion Network. [[code-tensorflow](https://github.com/TonythePlaneswalker/pcn)] [**`pc.`** ] :fire:

[[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460825)] Robust Generalized Point Cloud Registration Using Hybrid Mixture Model. [**`pc.`**]

[[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461049)] A General Framework for Flexible Multi-Cue Photometric Point Cloud Registration. [**`pc.`**]

[[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593839)] Dynamic Scaling Factors of Covariances for Accurate 3D Normal Distributions Transform Registration. [**`pc.`**]

[[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593558)] Robust Generalized Point Cloud Registration with Expectation Maximization Considering Anisotropic Positional Uncertainties. [**`pc.`**]

[[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594514)] PCAOT: A Manhattan Point Cloud Registration Method Towards Large Rotation and Small Overlap. [**`pc.`**]

[[IEEE Access](https://ieeexplore.ieee.org/document/8404075)] Multi-temporal Remote Sensing Image Registration Using Deep Convolutional Features [[code](https://github.com/yzhq97/cnn-registration)] :fire:  [**`rs.`**]




## 2017

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)] 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions. [[code](https://github.com/andyzeng/3dmatch-toolbox)] [**`pc.`** **`data.`** ] :fire: :star:  

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Vongkulbhisal_Discriminative_Optimization_Theory_CVPR_2017_paper.pdf)] Discriminative Optimization: Theory and Applications to Point Cloud Registration. [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Elbaz_3D_Point_Cloud_CVPR_2017_paper.pdf)] 3D Point Cloud Registration for Localization using a Deep Neural Network Auto-Encoder. [[code](https://github.com/gilbaz/LORAX)] [**`pc.`**]

[[CVPR](https://ieeexplore.ieee.org/document/8100078)] Convex Global 3D Registration with Lagrangian Duality.

[[CVPR](https://ieeexplore.ieee.org/document/8099746)] Group-Wise Point-Set Registration Based on Rényi's Second Order Entropy.

[[CVPR](https://ieeexplore.ieee.org/document/8100188)] Fine-to-Coarse Global Registration of RGB-D Scans.

[[CVPR](https://ieeexplore.ieee.org/document/8099652)] Joint Registration and Representation Learning for Unconstrained Face Identification.

[[CVPR](https://ieeexplore.ieee.org/document/8099970)] A General Framework for Curve and Surface Comparison and Registration with Oriented Varifolds.

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)] Colored Point Cloud Registration Revisited. [**`pc.`**]

[[ICCV](https://ieeexplore.ieee.org/document/8237364)] Local-to-Global Point Cloud Registration Using a Dictionary of Viewpoint Descriptors.

[[ICCV](https://ieeexplore.ieee.org/document/8237289)] Joint Layout Estimation and Global Multi-view Registration for Indoor Reconstruction.

[[ICCV](https://ieeexplore.ieee.org/document/8237718)] Deep Free-Form Deformation Network for Object-Mask Registration.

[[ICCV](https://ieeexplore.ieee.org/document/8237553)] Point Set Registration with Global-Local Correspondence and Transformation Estimation.

[[ICCV](https://ieeexplore.ieee.org/document/8237369)] Surface Registration via Foliation.

[[AAAI](https://arxiv.org/pdf/1611.10336.pdf)] An Artificial Agent for Robust Image Registration.

[[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/11195)] Non-Rigid Point Set Registration with Robust Transformation Estimation under Manifold Regularization.

[[ICRA](https://ieeexplore.ieee.org/document/7989664)] Using 2 point+normal sets for fast registration of point clouds with small overlap. [**`pc.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2016.2630687)] Image Registration and Change Detection under Rolling Shutter Motion Blur. 

[[TPAMI](https://doi.org/10.1109/TPAMI.2016.2567398)] Hyperbolic Harmonic Mapping for Surface Registration.

[[TPAMI](https://doi.org/10.1109/TPAMI.2016.2598344)] Randomly Perturbed B-Splines for Nonrigid Image Registration.



## before_2016

### 2016

[ECCV] [Fast Global Registration](https://www.researchgate.net/profile/Vladlen_Koltun/publication/305983982_Fast_Global_Registration/links/57a8086908aefe6167bc8366/Fast-Global-Registration.pdf) [Code](https://github.com/intel-isl/FastGlobalRegistration)

### 2015

[TPAMI] [Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration](https://arxiv.org/pdf/1605.03344.pdf) [Code](https://github.com/yangjiaolong/Go-ICP)

### 2009

[ICRA] [Fast point feature histograms (FPFH) for 3D registration](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf) 

[RSS] [Generalized-ICP](http://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf) 

### 1992

[TPAMI] [A method for registration of 3-D shapes](https://www.researchgate.net/publication/3191994_A_method_for_registration_of_3-D_shapes_IEEE_Trans_Pattern_Anal_Mach_Intell) 

### 1987

[TPAMI] [Least-squares fitting of two 3-D point sets](https://www.researchgate.net/publication/224378053_Least-squares_fitting_of_two_3-D_point_sets_IEEE_T_Pattern_Anal) 



---

# Learning Resources

Many thanks to [**yzhao062**](https://github.com/yzhao062/anomaly-detection-resources/commits?author=yzhao062) [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources). I followed his style to collect resources 

**This resources collect:**

- **Books & Academic Papers** 
- **Datasets**
- **Open-source and Commercial Libraries/Toolkits**
- **On-line Courses and Videos**
- **Key Conferences & Journals**



---



## Papers

### Overview & Survey Papers

#### Medical Image

1. A. Sotiras, et.al., [“Deformable medical image registration: A survey,”]( https://ieeexplore.ieee.org/document/6522524 ) 2013.

2. N. J. Tustison, et.al., [“Learning image-based spatial transformations via convolutional neural networks : A review,” ]( https://www.sciencedirect.com/science/article/abs/pii/S0730725X19300037 )2019.
3. G. Haskins,et.al. [“Deep Learning in Medical Image Registration: A Survey,” ]( https://arxiv.org/pdf/1903.02026.pdf )2019.

4. N. Tustison, et.al., [“Learning image-based spatial transformations via convolutional neural networks: A review,”]( https://www.sciencedirect.com/science/article/abs/pii/S0730725X19300037 )2019.



#### Others

- [Eurographics 2022] [A Survey of Non-Rigid 3D Registration](https://arxiv.org/pdf/2203.07858.pdf) 
- [arXiv 2021] [A comprehensive survey on point cloud registration](https://arxiv.org/pdf/2103.02690.pdf) 




### Key Algorithms



---

## Datasets & Competitions

### Datasets

#### Medical Image 

|                           Dataset                            | Number | Modality  |     Region     |     Format      |
| :----------------------------------------------------------: | :----: | :-------: | :------------: | :-------------: |
|             [DIRLAB](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/)              |   10   |  4D  CT   |      Lung      |      .img       |
|                          [LPBA40]()                          |   40   |  3D  MRI  |    T1 Brain    | .img+.hdr  .nii |
|        [IBSR18](https://www.nitrc.org/projects/ibsr/)        |   18   |  3D  MRI  |    T1 Brain    |    .img+.hdr    |
|             [EMPIRE](https://empire10.grand-challenge.org/)             |   30   |   4D CT   |      Lung      |    .mhd+.raw    |
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
|         [CIMA](https://www.kaggle.com/datasets/jirkaborovec/histology-cima-dataset)          | 108 |  2D   | lesions |     .png        |



#### Natural image

[**Indoor LiDAR-RGBD Scan Dataset**](http://redwood-data.org/indoor_lidar_rgbd/index.html)

[**ETH3D SLAM & Stereo Benchmarks**](https://www.eth3d.net/)

[**EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

[**ViViD : Vision for Visibility Dataset**](https://sites.google.com/view/dgbicra2019-vivid)

[**Apolloscape: Scene Parsing**]( http://apolloscape.auto/scene.html)

[**KITTI Visual Odometry dataset**](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

[**NCLT Dataset**](http://robots.engin.umich.edu/nclt/)

[**Oxford Robotcar Dataset**](https://robotcar-dataset.robots.ox.ac.uk/)



#### Remote Sensing 

[**ISPRS Benchmarks**](https://www.isprs.org/education/benchmarks.aspx)

[**HPatches**](https://github.com/hpatches/hpatches-dataset): The HPatches dataset was used as the basis for the local descriptor evaluation challenge that was presented in the Local Features: State of the Art, Open Problems and Performance Evaluation workshop during ECCV 2016.

[**The Zurich Urban Micro Aerial Vehicle Dataset**](http://rpg.ifi.uzh.ch/zurichmavdataset.html)

[**Zurich Summer Dataset**](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset)

[**Inria Aerial Image Labeling DataSet**](https://project.inria.fr/aerialimagelabeling/)

[**LANDSAT**](https://github.com/olivierhagolle/LANDSAT-Download)

[**NWPU-RESISC45**](https://figshare.com/articles/dataset/NWPU-RESISC45_Dataset_with_12_classes/16674166)

[**DOTA**](https://captain-whu.github.io/DOTA/dataset.html)

[**MUUFLGulfport**](https://github.com/GatorSense/MUUFLGulfport)



#### Point Cloud

**The Stanford 3D Scanning Repository**（斯坦福大学的3d扫描存储库）

http://graphics.stanford.edu/data/3Dscanrep/

这应该是做点云数据最初大家用最多的数据集，其中包含最开始做配准的Bunny、Happy Buddha、Dragon等模型。

[[Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/)] The Stanford 3D Scanning Repository. [**`pc.`**]



**Shapenet**

ShapeNet是一个丰富标注的大规模点云数据集，其中包含了55中常见的物品类别和513000个三维模型。

The KITTI Vision Benchmark Suite

链接：http://www.cvlibs.net/datasets/kitti/

这个数据集来自德国卡尔斯鲁厄理工学院的一个项目，其中包含了利用KIT的无人车平台采集的大量城市环境的点云数据集（KITTI），这个数据集不仅有雷达、图像、GPS、INS的数据，而且有经过人工标记的分割跟踪结果，可以用来客观的评价大范围三维建模和精细分类的效果和性能。



**Robotic 3D Scan Repository**

链接：http://kos.informatik.uni-osnabrueck.de/3Dscans/

这个数据集比较适合做SLAM研究，包含了大量的 Riegl 和 Velodyne 雷达数据



**佐治亚理工大型几何模型数据集**

链接：https://www.cc.gatech.edu/projects/large_models/



**PASCAL3D+**

链接：http://cvgl.stanford.edu/projects/pascal3d.html

包含了12中刚体分类，每一类超过了3000个实例。并且包含了对应的imageNet中每一类的图像。

**其他总结**

链接：https://github.com/timzhang642/3D-Machine-Learning

 

Other 

**[awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis#---datasets)**

[[UWA Dataset](https://drive.google.com/drive/folders/1_ZeEIBug_Wd5OyWHlxQZACYCOh1dbB12)]  [**`pc.`**] (Uploaded by @sukun1045 for their repository [shlizee/Predict-Cluster](https://github.com/shlizee/Predict-Cluster))

[[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community  with the aim to facilitate result evaluations and comparisons. [ **`pc.`** ]

[[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. [**`pc.`** ]



### Competitions

#### CVPR

##### 2024

[Image Matching Challenge 2024](https://www.kaggle.com/competitions/image-matching-challenge-2024/overview)

##### 2023

[Image Matching Challenge 2023](https://www.kaggle.com/competitions/image-matching-challenge-2023/overview)

##### 2022

[Image Matching Challenge 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022/overview)

##### 2021

[Image Matching Challenge 2021](https://www.cs.ubc.ca/research/image-matching-challenge/current/)

##### 2020

[The Visual Localization Benchmark](https://www.visuallocalization.net/)



#### [All Challenges](https://grand-challenge.org/challenges/)

##### 2024

[Learn2Reg](https://learn2reg.grand-challenge.org/)

##### 2023

[Learn2reg](https://learn2reg.grand-challenge.org/learn2reg-2023/)

##### 2022

[ACROBAT](https://acrobat.grand-challenge.org/)

> MICCAI 2022

> the AutomatiC Registration Of Breast cAncer Tissue (ACROBAT) challenge

##### 2021

[Learn2Reg](https://learn2reg.grand-challenge.org/Learn2Reg2021/)

##### 2020

[Learn2Reg](https://learn2reg.grand-challenge.org/Learn2Reg2020/)

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

[EMPIRE](https://empire10.grand-challenge.org/)



---

## Toolbox

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

[c++] [**Elastix**](https://elastix.lumc.nl/):  open source software, based on the well-known [ITK](http://www.itk.org) . 

The software consists of a collection of algorithms that are commonly used to solve (medical) image registration problems.  [**[manual]**](https://elastix.lumc.nl/download/elastix-5.1.0-manual.pdf)

[C++] [Python] [Java] [R] [Ruby] [Lua] [Tcl] [C#] [**SimpleElastix**](http://simpleelastix.github.io/): a medical image registration library that makes  state-of-the-art image registration really easy to do in languages like  Python, Java and R. 

[**3D slicer**](https://www.slicer.org/) :  an open source software platform for  medical image informatics, image processing, and three-dimensional  visualization. Built over two decades through support from the  National Institutes of Health and a worldwide developer community, Slicer brings free, powerful cross-platform processing tools to  physicians, researchers, and the general public.  



**Github repository for deep learning medical image registration**:

 [Keras] [**VoxelMorph**](https://github.com/voxelmorph/voxelmorph) :fire:

 [Keras] [**FAIM**]( https://github.com/dykuang/Medical-image-registration ) :fire:

 [Tensorflow] [**Weakly-supervised CNN**](https://github.com/YipengHu/label-reg) :fire:

 [Tensorflow] [**RegNet3D** ](https://github.com/hsokooti/RegNet) :fire:

 [Tensorflow] [**Recursive-Cascaded-Networks**](https://github.com/microsoft/Recursive-Cascaded-Networks) 

 [Pytorch] [**Probabilistic Dense Displacement Network**](https://github.com/multimodallearning/pdd_net)

 [Pytorch] [**Linear and Deformable Image Registration**](https://github.com/shreshth211/image-registration-cnn)

 [Pytorch] [**Inverse-Consistent Deep Networks**](https://github.com/zhangjun001/ICNet) 

 [Pytorch] [**Non-parametric image registration**](https://github.com/uncbiag/registration) :fire:

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

[Parallax-tolerant image stitching](https://github.com/gain2217/Robust_Elastic_Warping) :fire:



### Point Cloud

#### MeshLab

> 简介：是一款开源、可移植和可扩展的三维几何处理系统。主要用于处理和编辑3D三角网格，它提供了一组用于编辑、清理、修复、检查、渲染、纹理化和转换网格的工具。提供了处理由3D数字化工具/设备生成的原始数据以及3D打印功能，功能全面而且丰富。MeshLab支持多数市面上常见的操作系统，包括Windows、Linux及Mac OS X，支持输入/输出的文件格式有：STL 、OBJ 、 VRML2.0、U3D、X3D、COLLADA
> MeshLab可用于各种学术和研究环境，如微生物学、文化遗产及表面重建等。

#### ICP开源库

[SLAM6D](http://slam6d.sourceforge.net/)

[Libicp](http://www.cvlibs.net/software/libicp/)

[libpointmatcher ](https://github.com/ethz-asl/libpointmatcher) :fire:

[g-icp](https://github.com/avsegal/gicp) :fire:

[n-icp](http://jacoposerafin.com/nicp/)

---


## Books & Tutorials    

### Books

#### Natural image

[Multiple view geometry in computer vision](https://www.robots.ox.ac.uk/~vgg/hzbook/) by Richard Hartley and Andrew Zisserman, 2004: Mathematic and geometric basis for 2D-2D and 2D-3D registration. A **must-read** for people in the field of registration. [E-book](https://github.com/DeepRobot2020/books/blob/master/Multiple View Geometry in Computer Vision (Second Edition).pdf)

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



### Tutorials

#### Natural image

- **[ImageRegistration](https://github.com/quqixun/ImageRegistration)** :fire:

A demo that implement image registration by matching SIFT descriptors and appling RANSAC and affine transformation.

#### Medical Image

- [**Medical Image Registration**](https://github.com/natandrade/Tutorial-Medical-Image-Registration) :fire:

- [MICCAI2019] [**learn2reg**](https://github.com/learn2reg/tutorials2019) [PDF](https://github.com/learn2reg/tutorials2019/blob/master/slides) :fire:

> Big thanks to [Yipeng Hu]( https://github.com/YipengHu ) organizing the excellent tutorial.
>
> **Description:**
>
> Medical image registration has been a cornerstone in the research fields of medical image computing and computer assisted intervention, responsible for many clinical applications. Whilst machine learning methods have long been important in developing pairwise algorithms, recently proposed deep-learning-based frameworks directly infer displacement fields without iterative optimization for unseen image pairs, using neural networks trained from large population data. These novel approaches promise to tackle several most challenging aspects previously faced by classical pairwise methods, such as high computational cost, robustness for generalization and lack of inter-modality similarity measures. 
>
> Output from several international research groups working in this area include award-winning conference presentations, high-impact journal publications, well-received open-source implementations and industrial-partnered translational projects, generating significant interests to all levels of world-wide researchers. Accessing to the experience and expertise in this inherently multidisciplinary topic can be beneficial to many in our community, especially for the next generation of young scientists, engineers and clinicians who often have only been exposed to a subset of these methodologies and applications. 
>
> We organize a tutorial including both theoretical and practical sessions, inviting expert lectures and tutoring coding for real-world examples. Three hands-on sessions guiding participants to understand and implement published algorithms using clinical imaging data. This aims to provide an opportunity for the participants to bridge the gap between expertises in medical image registration and deep learning, as well as to start a forum to discuss know-hows, challenges and future opportunities in this area.

- [MICCAI2019] [**Autograd Image Registration Laboratory**](https://github.com/airlab-unibas/MICCAITutorial2019)
- [kaggle 2018] [**X-Ray Patient Scan Registration**](https://www.kaggle.com/kmader/x-ray-patient-scan-registration)

> SimpleITK, ITK, scipy, OpenCV, Tensorflow and PyTorch all offer tools for registering images, we explore a few here to see how well they work when applied to the fairly tricky problem of registering from the same person at different time and disease points.

- [Sibgrapi 2018] **Practical Review on Medical Image Registration: from Rigid to Deep Learning based Approaches** [[PDF&Slides\]](https://github.com/natandrade/Tutorial-Medical-Image-Registration) :fire:

> A tutorial for anyone who wants to learn Medical Image Registration, by  Natan Andrade, Fabio Augusto Faria, Fábio Augusto Menocci Cappabianco

- [kaggle 2016] [**Image registration, the R way, (almost) from scratch**](https://www.kaggle.com/vicensgaitan/image-registration-the-r-way)

> There are some packages in R for image manipulation and after some test I select “imager” , based on the CImg C++, fast and providing several image processing tools.

- [MIT] [**HST.582J**](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-582j-biomedical-signal-and-image-processing-spring-2007/)  Biomedical Signal and Image Processing [PDF](https://ocw.mit.edu/courses/health-sciences-and-technology/hst-582j-biomedical-signal-and-image-processing-spring-2007/lecture-notes/l16_reg1.pdf) 



#### Remote Sensing
- [Image Alignment and Stitching: A Tutorial](http://szeliski.org/papers/Szeliski_ImageAlignment_MSR-TR-2004-92.pdf)

- [Image Registration for Remote Sensing](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120008278.pdf)

- [Image Stitching](https://www.zhihu.com/question/34535199/answer/135169187)

- [The Remote Sensing Tutorial 1](https://www.ucl.ac.uk/EarthSci/people/lidunka/GEOL2014/Geophysics%2010%20-Remote%20sensing/Remote%20Sensing%20Tutorial%20Overview.htm)

- [The Remote Sensing Tutorial 2](https://www.nrcan.gc.ca/maps-tools-publications/satellite-imagery-air-photos/tutorial-fundamentals-remote-sensing/9309)
  
#### Point Cloud

- [点云配准算法说明与流程介绍](https://blog.csdn.net/Ha_ku/article/details/79755623)

- [点云配准算法介绍与比较](https://blog.csdn.net/weixin_43236944/article/details/88188532)

- [机器学习方法处理三维点云](https://blog.csdn.net/u014636245/article/details/82755966)

- [一个例子详细介绍点云配准的过程](https://www.zhihu.com/question/34170804/answer/121533317)



### Blogs

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

[Image Registration: From SIFT to Deep Learning](https://blog.csdn.net/leoking01/article/details/115540817)



#### 点云配准

[点云配准算法的说明与流程介绍](https://blog.csdn.net/Ha_ku/article/details/79755623)

[几种点云配准算法的方法的介绍与比较](https://blog.csdn.net/weixin_43236944/article/details/88188532)

[三维点云用机器学习的方法进行处理](https://blog.csdn.net/u014636245/article/details/82755966)

[一个例子详细介绍了点云配准的过程](https://www.zhihu.com/question/34170804)

---

## Courses Seminars and Videos

### Courses

[**16-822: Geometry-based Methods in Vision**](http://www.cs.cmu.edu/~hebert/geom.html)

[VALSE 2018] [Talk: 2017以来的2D to 3D](https://zhuanlan.zhihu.com/p/38611920) by 吴毅红



### Workshops

[CVPR 2021 Image Matching: Local Features and Beyond](https://image-matching-workshop.github.io/)



[WBIR - International Workshop on Biomedical Image Registration](https://dblp.org/db/conf/wbir/index.html)

> [WBIR 2022](https://www.mevis.fraunhofer.de/en/fairs-and-conferences/2022/wbir-2022.html): Munich, Germany
>
> [WBIR 2020](https://wbir2020.org/): Portorož, Slovenia
>
> [WBIR 2018](https://wbir2018.nl/index.html): Leiden, Netherlands
>
> [WBIR 2016](http://wbir2016.doc.ic.ac.uk/): Las Vegas NV 
>
> [WBIR 2014](http://wbir2014.cs.ucl.ac.uk/): London, UK  



### Seminars



### Videos

- [Definition and Introduction to Image Registration Pre Processing Overview](https://www.youtube.com/watch?v=sGNFmAGqpZ8)

- [仿射变换与图像配准](https://www.bilibili.com/video/av52733294)（科普性视频， 比较简陋）

#### Remote Sensing
- [Registration of images of different modalities in Remote Sensing](https://youtu.be/9pPwNN-7oWU)



---


## Key Conferences/Workshops/Journals

### Conferences and Workshops

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

[**ICRA**](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/): IEEE International Conference on Robotics and Automation

[**International Conference on 3D Vision**](http://3dv19.gel.ulaval.ca/)

[**WACV**](https://wacv20.wacv.net/): Winter Conference on Applications of Computer Vision



#### Biomedical image

[**MICCAI**]( http://www.miccai.org/ ): International Conference on Medical Image Computing and Computer Assisted Intervention

[**IPMI**](http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=77376): Information Processing in Medical Imaging

[**ISBI**]( http://2020.biomedicalimaging.org/): International Symposium on Biomedical Imaging 

[**Medical Imaging SPIE**](https://spie.org/conferences-and-exhibitions/medical-imaging?SSO=1 )



#### Remote Sensing

[ISPRS-2020](http://www.isprs2020-nice.com/)



#### Point Cloud

点云配准主要应用于工业制造业的逆向工程、古文物修复、医学三维图像构建等领域。研究内容是属于计算机视觉领域的研究范畴。国际上的会议如计算机视觉三大顶会ICCV、CVPR、ECCV等都会有相关技术，除此之外，还有ACCV、BMVC、SSVM等认可度也比较高。



### Journals

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



#### Point Cloud

IEEE旗下的TPAMI，TIP等，还有SIAM Journal Image Sciences，Springer那边有IJCV





---

## **How to contact us**

We have QQ Group  【配准萌新交流群】（**已满3000人**）869211738  欢迎加入【配准萌新交流（二群)】 929134506 and Wechat Group 【配准交流群】（**已满员**） for comunications.



**More items will be added to the repository**.
Please feel free to suggest other key resources by opening an issue report,
submitting a pull request, or dropping me an email @ (im.young@foxmail.com).
Enjoy reading!







## Acknowledgments

Many thanks :heart: to all project contributors:

[![](https://opencollective.com/awesome-image-registration/contributors.svg?width=890&button=false)](https://github.com/Awesome-Image-Registration-Organization/awesome-image-registration/graphs/contributors)



Many thanks :heart: to the other awesome list:

- **[Yochengliu](https://github.com/Yochengliu)**  [awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis) 
- **[zhulf0804](https://github.com/zhulf0804)**  [3D-PointCloud](https://github.com/zhulf0804/3D-PointCloud)
- **[NUAAXQ](https://github.com/NUAAXQ)**  [awesome-point-cloud-analysis-2022](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2022)  
- [**visionxiang**](https://github.com/visionxiang)   [awesome-computational-photography](https://github.com/visionxiang/awesome-computational-photography)
- [**tzxiang**](https://github.com/tzxiang)  [awesome-image-alignment-and-stitching](https://github.com/tzxiang/awesome-image-alignment-and-stitching)  
- **[hoya012](https://github.com/hoya012)**  [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)     
- [**JunMa11**](https://github.com/JunMa11) [MICCAI-OpenSourcePapers](https://github.com/JunMa11/MICCAI-OpenSourcePapers)
- **[Amusi](https://github.com/amusi)**  [awesome-object-detection](https://github.com/amusi/awesome-object-detection) 
- [**youngfish42**](https://github.com/youngfish42)  [Awesome-Federated-Learning-on-Graph-and-Tabular-Data](https://github.com/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data)  
- [**yzhao062**](https://github.com/yzhao062/)  [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources)
