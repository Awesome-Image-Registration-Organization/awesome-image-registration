# Awesome Image Registration

A curated list of image registration related books, papers, videos, and toolboxes 

[![Stars](https://img.shields.io/github/stars/youngfish42/image-registration-resources.svg?color=orange)](https://github.com/youngfish42/image-registration-resources/stargazers)  [![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86%E6%8C%87%E5%8C%97-blue)](https://zhuanlan.zhihu.com/Image-Registration)  [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re) [![License](https://img.shields.io/github/license/youngfish42/image-registration-resources.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) 

[**Image registration**](https://en.wikipedia.org/wiki/Image_registration) is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, and from different sensors, times, depths, or viewpoints.

It is used in computer vision, medical imaging, military automatic target recognition, compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from different measurements. 

We use another project to automatically track updates to IR papers, click on [IR-paper-update-tracker](https://github.com/Awesome-Image-Registration-Organization/IR-paper-update-tracker) if you need it.

Please note that if this page does not display the full content, please visit the [official homepage](https://awesome-image-registration-organization.github.io/awesome-image-registration/) for full information.

[**Paper Lists**](#Paper-Lists) 

- [2025](#2025)
- [2024](#2024)
- [2023](#2023)
- [2022](#2022)
- [2021](#2021)
- [2020](#2020)
- [2019](#2019)
- [2018](#2018)
- [2017](#2017)

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

*Last updated: 2025/09/21*

*2025/09/21* - update recent papers

*2025/04/25* - update recent papers and add the repository link of [Awesome-Medical-Image-Registration](https://github.com/Alison-brie/Awesome-Medical-Image-Registration)

*2024/12/03* - update recent papers

*2024/04/30* - update recent papers on [TPAMI](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AIEEE_Trans._Pattern_Anal._Mach._Intell.%3A)/[MICCAI](https://dblp.org/search?q=registra%20venue%3AMICCAI%3A)/[CVPR](https://dblp.org/search?q=registra%20%20venue%3ACVPR%3A)/[ICCV](https://dblp.org/search?q=registra%20venue%3AICCV%3A)/[ECCV](https://dblp.org/search?q=registra%20venue%3AECCV%3A)/[AAAI](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3AAAAI%3A)/[NeurIPS](https://dblp.org/search?q=registra%20venue%3ANeurIPS%3A)/[MIA](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AMedical_Image_Anal.%3A)/[ICLR](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3AICLR%3A)

*2023/03/02* - add papers according to [3D-PointCloud](https://github.com/zhulf0804/3D-PointCloud), update recent papers on CVPR/ECCV 2022

*2022/07/27* - update recent papers on ECCV 2022

*2022/07/14* - add the corresponding open source code for the 2022 papers

*2022/07/12* - update recent papers on AAAI 2022 and add information about competitions

*2022/06/19* - update recent TPAMI papers (2017-2021) about image registration according to dblp search engine, update recent MICCAI papers  about image registration according to  [MICCAI-OpenSourcePapers](https://github.com/JunMa11/MICCAI-OpenSourcePapers).

*2022/06/18* - update recent papers (2017-2021) on CVPR/ICCV/ECCV/AAAI/NeurIPS/MIA about image registration according to dblp search engine.

*2022/06/18* - update papers (2020-2022) about point cloud registration from [awesome-point-cloud-analysis-2023](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2023).

*2020/04/20* - update recent papers (2017-2020) about point cloud registration and make some diagram about history of image registration.

## 2025

### MICCAI
- Guiding Registration with Emergent Similarity from Pre-Trained Diffusion Models. [[PUB](https://arxiv.org/pdf/2506.02419)] [CODE](https://github.com/uncbiag/dgir)
- Spatial regularisation for improved accuracy and interpretability in keypoint-based registration. [[PUB](https://arxiv.org/pdf/2503.04499)] [CODE](https://github.com/BenBillot/spatial_regularisation)
- Mono-Modalizing Extremely Heterogeneous Multi-Modal Medical Image Registration. [[PUB](https://arxiv.org/abs/2506.15596)] [CODE](https://github.com/MICV-yonsei/M2M-Reg)
- VoxelOpt: Voxel-Adaptive Message Passing for Discrete Optimization in Deformable Abdominal CT Registration. [[PUB](https://arxiv.org/pdf/2506.19975)] [CODE](https://github.com/tinymilky/VoxelOpt)
- New Multimodal Similarity Measure for Image Registration via Modeling Local Functional Dependence with Linear Combination of Learned Basis Functions. [[PUB](https://arxiv.org/pdf/2503.05335)]
- A Novel Streamline-based diffusion MRI Tractography Registration Method with Probabilistic Keypoint Detection. [[PUB](https://arxiv.org/pdf/2503.02481)]
- Implicit Deformable Medical Image Registration with Learnable Kernels. [[PUB](https://www.arxiv.org/pdf/2506.02150)]
- Bridging Classical and Learning-based Iterative Registration through Deep Equilibrium Models. [[PUB](https://arxiv.org/pdf/2507.00582)]
- Deformable Registration Framework for Augmented Reality-based Surgical Guidance in Head and Neck Tumor Resection. [[PUB](https://arxiv.org/pdf/2503.08802)]
- Vascular Photoacoustic Volume Registration via 2D Feature Matching with Reverse Mapping Based on Maximum Intensity Projection. [[PUB](https://papers.miccai.org/miccai-2025/paper/0431_paper.pdf)]
- PromptReg: Universal Medical Image Registration via Task Prompt Learning and Domain Knowledge Transfer. [[PUB](https://papers.miccai.org/miccai-2025/paper/1233_paper.pdf)]
- EUReg: End-to-end Framework for Efficient 2D-3D Ultrasound Registration. [[PUB](https://papers.miccai.org/miccai-2025/paper/1387_paper.pdf)]
- EG-Net: An Edge-Guided Network for Rigid Registration of Laparoscopic Low-Overlap Point Clouds. [[PUB](https://papers.miccai.org/miccai-2025/paper/1387_paper.pdf)]
- DGMIR: Dual-Guided Multimodal Medical Image Registration based on Multi-view Augmentation and On-site Modality Removal. [[PUB](https://papers.miccai.org/miccai-2025/paper/1691_paper.pdf)]
- Weakly-Supervised 2D/3D Image Registration via Differentiable X-ray Rendering and ROI Segmentation. [[PUB](https://papers.miccai.org/miccai-2025/paper/2022_paper.pdf)]
- RDMR: Recursive Inference and Representation Disentanglement for Multimodal Large Deformation Registration. [[PUB](https://papers.miccai.org/miccai-2025/paper/2376_paper.pdf)]
- LDDMEm: Large Deformation Diffeomorphic Metric Embedding Decoupling Shape Analysis from Image Registration. [[PUB](https://papers.miccai.org/miccai-2025/paper/3223_paper.pdf)]
- RadGS-Reg: Registering Spine CT with Biplanar X-rays via Joint 3D Radiative Gaussians Reconstruction and 3D/3D Registration. [[PUB](https://arxiv.org/pdf/2508.21154)]
- Towards Patient-Specific Deformable Registration in Laparoscopic Surgery. [[PUB](https://papers.miccai.org/miccai-2025/paper/3372_paper.pdf)]
- Structure-Preserve Expansion for Medical Image Registration with Minimal Overlap. [[PUB]()]
- Gaussian Primitive Optimized Deformable Retinal Image Registration. [[PUB](https://arxiv.org/pdf/2508.16852)]
- Probabilistic Inverse Consistent Image Registration Using Sparse Bayesian Network. [[PUB](https://papers.miccai.org/miccai-2025/paper/4995_paper.pdf)]

### AAAI
- Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/32251)]
- HybridReg: Robust 3D Point Cloud Registration with Hybrid Motions. [[PUB](https://doi.org/10.1609/aaai.v39i3.32284)]
- PSReg: Prior-guided Sparse Mixture of Experts for Point Cloud Registration. [[PUB](https://doi.org/10.1609/aaai.v39i4.32395)]
- GRICP: Granular-Ball Iterative Closest Point with Multikernel Correntropy for Point Cloud Fine Registration. [[PUB](https://doi.org/10.1609/aaai.v39i2.32164)]
- Where Precision Meets Efficiency: Transformation Diffusion Model for Point Cloud Registration. [[PUB](https://doi.org/10.1609/aaai.v39i9.33055)]
- Partial Point Cloud Registration with Multi-view 2D Image Learning. [[PUB](https://doi.org/10.1609/aaai.v39i10.33121)]
- A Gaussian Filter-Based 3D Registration Method for Series Section Electron Microscopy. [[PUB](https://doi.org/10.1609/aaai.v39i1.32103)]
- Cross-PCR: A Robust Cross-Source Point Cloud Registration Framework. [[PUB](https://doi.org/10.1609/aaai.v39i10.33129)]

### ICLR
- Learning General-purpose Biomedical Volume Representations using Randomized Synthesis. [[PUB](https://arxiv.org/abs/2411.02372)] [CODE](https://github.com/neel-dey/anatomix)
- Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy. [[PUB](https://openreview.net/forum?id=cjJqU40nYS)]
- Hierarchical Uncertainty Estimation for Learning-based Registration in Neuroimaging. [[PUB](https://openreview.net/forum?id=w8LMtFY97b)]

### CVPR
- SACB-Net: Spatial-awareness Convolutions for Medical Image Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_SACB-Net_Spatial-awareness_Convolutions_for_Medical_Image_Registration_CVPR_2025_paper.html)] [CODE](https://github.com/x-xc/SACB_Net)
- CARL: A Framework for Equivariant Image Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Greer_CARL_A_Framework_for_Equivariant_Image_Registration_CVPR_2025_paper.html)]
- MultiMorph: On-demand Atlas Construction. [[PUB](https://arxiv.org/pdf/2504.00247)] [CODE](https://github.com/mabulnaga/multimorph)
- GraphI2P: Image-to-Point Cloud Registration with Exploring Pattern of Correspondence via Graph Learning. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Bie_GraphI2P_Image-to-Point_Cloud_Registration_with_Exploring_Pattern_of_Correspondence_via_CVPR_2025_paper.html)]
- ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Edstedt_ColabSfM_Collaborative_Structure-from-Motion_by_Point_Cloud_Registration_CVPR_2025_paper.html)]
- Dual Focus-Attention Transformer for Robust Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_Dual_Focus-Attention_Transformer_for_Robust_Point_Cloud_Registration_CVPR_2025_paper.html)]
- Zero-shot RGB-D Point Cloud Registration with Pre-trained Large Vision Model. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Zero-shot_RGB-D_Point_Cloud_Registration_with_Pre-trained_Large_Vision_Model_CVPR_2025_paper.html)]
- Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.html)]
- Implicit Correspondence Learning for Image-to-Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Implicit_Correspondence_Learning_for_Image-to-Point_Cloud_Registration_CVPR_2025_paper.html)]
- AutoURDF: Unsupervised Robot Modeling from Point Cloud Frames Using Cluster Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_AutoURDF_Unsupervised_Robot_Modeling_from_Point_Cloud_Frames_Using_Cluster_CVPR_2025_paper.html)]
- Stable-SCore: A Stable Registration-based Framework for 3D Shape Correspondence. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Stable-SCore_A_Stable_Registration-based_Framework_for_3D_Shape_Correspondence_CVPR_2025_paper.html)]
- Cross-Rejective Open-Set SAR Image Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Mao_Cross-Rejective_Open-Set_SAR_Image_Registration_CVPR_2025_paper.html)]
- HeMoRa: Unsupervised Heuristic Consensus Sampling for Robust Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HeMoRa_Unsupervised_Heuristic_Consensus_Sampling_for_Robust_Point_Cloud_Registration_CVPR_2025_paper.html)]
- Unlocking Generalization Power in LiDAR Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Zeng_Unlocking_Generalization_Power_in_LiDAR_Point_Cloud_Registration_CVPR_2025_paper.html)]
- Progressive Correspondence Regenerator for Robust 3D Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Progressive_Correspondence_Regenerator_for_Robust_3D_Registration_CVPR_2025_paper.html)]

### TPAMI
- Homeomorphism Prior for False Positive and Negative Problem in Medical Image Dense Contrastive Representation Learning. [[PUB](https://www.arxiv.org/abs/2502.05282)] [CODE](https://github.com/YutingHe-list/GEMINI)

### MIA
- PViT-AIR: Puzzling vision transformer-based affine image registration for multi histopathology and faxitron images of breast tissue. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002810)]
- MUsculo-Skeleton-Aware (MUSA) deep learning for anatomically guided head-and-neck CT deformable registration. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002767)]
- USLR: An open-source tool for unbiased and smooth longitudinal registration of brain MRI. [[PUB](https://doi.org/10.1016/j.media.2025.103662)]
- Nested hierarchical group-wise registration with a graph-based subgrouping strategy for efficient template construction. [[PUB](https://doi.org/10.1016/j.media.2025.103624)]
- A survey on deep learning in medical image registration: New technologies, uncertainty, evaluation metrics, and beyond. [[PUB](https://doi.org/10.1016/j.media.2024.103385)]
- Deep implicit optimization enables robust learnable features for deformable image registration. [[PUB](https://doi.org/10.1016/j.media.2025.103577)]
- Domain agnostic 2D-3D deformable registration Application to fluoroscopic guidance without contrast agent. [[PUB](https://doi.org/10.1016/j.media.2025.103688)]
- FPM-R2Net: Fused Photoacoustic and operating Microscopic imaging with cross-modality Representation and Registration Network. [[PUB](https://doi.org/10.1016/j.media.2025.103698)]
- SynMSE: A multimodal similarity evaluator for complex distribution discrepancy  in unsupervised deformable multimodal medical image registration. [[PUB](https://doi.org/10.1016/j.media.2025.103620)]

## 2024

### AAAI
- Test-Time Adaptation via Style and Structure Guidance for Histological Image Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/28601)]
- PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/27775)] [**`pc.`**]
- SuperJunction: Learning-Based Junction Detection for Retinal Image Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/27782)]
- SPEAL: Skeletal Prior Embedded Attention Learning for Cross-Source Point Cloud Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/28446)] [**`pc.`**]

### CVPR
- Dynamic Cues-Assisted Transformer for Robust Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10655264)]
- Bi-level Learning of Task-Specific Decoders for Joint Registration and One-Shot Medical Image Segmentation. [[PUB](https://ieeexplore.ieee.org/document/10656811)]
- H-ViT: A Hierarchical Vision Transformer for Deformable Image Registration. [[PUB](https://ieeexplore.ieee.org/document/10656382)]
- Intraoperative 2D/3D Image Registration via Differentiable X-Ray Rendering. [[PUB](https://ieeexplore.ieee.org/document/10655316)]
- Scalable 3D Registration via Truncated Entry-Wise Absolute Residuals. [[PUB](https://ieeexplore.ieee.org/document/10656079)]
- Diffeomorphic Template Registration for Atmospheric Turbulence Mitigation. [[PUB](https://ieeexplore.ieee.org/document/10658095)]
- Extend Your Own Correspondences: Unsupervised Distant Point Cloud Registration by Progressive Distance Extension. [[PUB](https://ieeexplore.ieee.org/document/10657717)]
- IIRP-Net: Iterative Inference Residual Pyramid Network for Enhanced Image Registration. [[PUB](https://ieeexplore.ieee.org/document/10657659)]
- Correlation-aware Coarse-to-fine MLPs for Deformable Medical Image Registration. [[PUB](https://ieeexplore.ieee.org/document/10657635)]
- Modality-Agnostic Structural Image Representation Learning for Deformable Multi-Modality Medical Image Registration. [[PUB](https://ieeexplore.ieee.org/document/10656271)]
- ColorPCR: Color Point Cloud Registration with Multi-Stage Geometric-Color Fusion. [[PUB](https://ieeexplore.ieee.org/document/10655149)]
- From a Bird's Eye View to See: Joint Camera and Subject Registration without the Camera Calibration. [[PUB](https://ieeexplore.ieee.org/document/10657684)]
- Learning Instance-Aware Correspondences for Robust Multi-Instance Point Cloud Registration in Cluttered Scenes. [[PUB](https://ieeexplore.ieee.org/document/10655087)]
- Inlier Confidence Calibration for Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10657889)]
- Correspondence-Free Non-Rigid Point Set Registration Using Unsupervised Clustering Analysis. [[PUB](https://ieeexplore.ieee.org/document/10656571)]

### ECCV
- GaussReg: Fast 3D Registration with Gaussian Splatting. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72633-0_23)]
- PointRegGPT: Boosting 3D Point Cloud Registration Using Generative Point-Cloud Pairs for Training. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72983-6_16)]
- SemReg: Semantics Constrained Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72940-9_17)]
- Unsupervised Multi-modal Medical Image Registration via Invertible Translation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72751-1_2)]
- UMERegRobust - Universal Manifold Embedding Compatible Features for Robust Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73016-0_21)]
- Equi-GSPR: Equivariant SE(3) Graph Network Model for Sparse Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73235-5_9)]
- NICP: Neural ICP for 3D Human Registration at Scale. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73636-0_16)]
- Fast Registration of Photorealistic Avatars for VR Facial Animation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73033-7_23)]
- Weakly-Supervised Camera Localization by Ground-to-Satellite Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72673-6_3)]
- NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73223-2_13)]
- Diff-Reg: Diffusion Model in Doubly Stochastic Matrix Space for Registration Problem. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73650-6_10)]
- ML-SemReg: Boosting Point Cloud Registration with Multi-level Semantic Consistency. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72784-9_2)]
- PARE-Net: Position-Aware Rotation-Equivariant Networks for Robust Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72904-1_17)]
- Adaptive Correspondence Scoring for Unsupervised Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72920-1_5)]
- Correspondence-Free SE(3) Point Cloud Registration in RKHS via Unsupervised Equivariant Learning. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-73223-2_5)]

### TPAMI
- Match Normalization: Learning-Based Point Cloud Registration for 6D Object Pose Estimation in the Real World. [[PUB](https://ieeexplore.ieee.org/document/10402084/)]
- Efficient and Robust Point Cloud Registration via Heuristics-Guided Parameter Search. [[PUB](https://ieeexplore.ieee.org/document/10496861)]
- Transformation Decoupling Strategy Based on Screw Theory for Deterministic Point Cloud Registration With Gravity Prior. [[PUB](https://ieeexplore.ieee.org/document/10634827)]
- MAC: Maximal Cliques for 3D Registration. [[PUB](https://ieeexplore.ieee.org/document/10636064)]
- RIGA: Rotation-Invariant and Globally-Aware Descriptors for Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10380454)] [**`pc.`**]

### MICCAI
- Keypoint Matching for Instrument-Free 3D Registration in Video-Based Surgical Navigation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_32)]
- WiNet: Wavelet-Based Incremental Learning for Efficient Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_71)]
- Intraoperative Registration by Cross-Modal Inverse Neural Rendering. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_30)]
- One Registration is Worth Two Segmentations. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_62)]
- Epicardium Prompt-Guided Real-Time Cardiac Ultrasound Frame-to-Volume Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_58)]
- GMM-CoRegNet: A Multimodal Groupwise Registration Framework Based on Gaussian Mixture Model. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_59)]
- Biomechanics-Informed Non-rigid Medical Image Registration and its Inverse Material Property Estimation with Linear and Nonlinear Elasticity. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_53)]
- Data-Driven Tissue- and Subject-Specific Elastic Regularization for Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_54)]
- Hierarchical Symmetric Normalization Registration Using Deformation-Inverse Network. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_62)]
- PULPo: Probabilistic Unsupervised Laplacian Pyramid Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_67)]
- DINO-Reg: General Purpose Image Encoder for Training-Free Multi-modal Deformable Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_57)]
- uniGradICON: A Foundation Model for Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_70)]
- LIBR+: Improving Intraoperative Liver Registration by Learning the Residual of Biomechanics-Based Deformable Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72089-5_34)]
- Groupwise Deformable Registration of Diffusion Tensor Cardiovascular Magnetic Resonance: Disentangling Diffusion Contrast, Respiratory and Cardiac Motions. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_60)]
- Toward Universal Medical Image Registration via Sharpness-Aware Meta-Continual Learning. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_69)]
- TLRN: Temporal Latent Residual Networks for Large Deformation Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_68)]
- Noise Removed Inconsistency Activation Map for Unsupervised Registration of Brain Tumor MRI Between Pre-operative and Follow-Up Phases. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_64)]
- On-the-Fly Guidance Training for Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_65)]
- MemWarp: Discontinuity-Preserving Cardiac Registration with Memorized Anatomical Filters. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_63)]
- Heteroscedastic Uncertainty Estimation Framework for Unsupervised Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_61)]
- Deep-Learning-Based Groupwise Registration for Motion Correction of Cardiac T1 Mapping. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_55)]
- DiffuseReg: Denoising Diffusion Model for Obtaining Deformation Fields in Unsupervised Deformable Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_56)]

### MIA
- Placental vessel segmentation and registration in fetoscopy: Literature review and MICCAI FetReg2021 challenge findings. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523003262)]
- Classification, registration and segmentation of ear canal impressions using convolutional neural networks. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S136184152400077X)]
- JOSA: Joint surface-based registration and atlas construction of brain geometry and function. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002172)]
- Automatic registration with continuous pose updates for marker-less surgical navigation in spine surgery. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002876)]
- CMAN: Cascaded Multi-scale Spatial Channel Attention-guided Network for large 3D deformable registration of liver CT images. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001373)]
- SUGAR: Spherical ultrafast graph attention framework for cortical surface registration. [[PUB](https://www.sciencedirect.com/science/article/pii/S1361841524000471)]
- Comparing regularized Kelvinlet functions and the finite element method for registration of medical images to sparse organ data. [[PUB](https://www.sciencedirect.com/science/article/pii/S1361841524001464)]
- Medical image registration via neural fields. [[PUB](https://www.sciencedirect.com/science/article/pii/S1361841524001749)]
- Privacy preserving image registration. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000549)]
- Retinal image registration method for myopia development. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001671)]
- The ACROBAT 2022 challenge: Automatic registration of breast cancer tissue. [[PUB](https://www.sciencedirect.com/science/article/pii/S1361841524001828)]
- PRSCS-Net: Progressive 3D/2D rigid Registration network with the guidance of Single-view Cycle Synthesis. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S1361841524002081)]
- Longitudinally consistent registration and parcellation of cortical surfaces using semi-supervised learning. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S136184152400118X)]
- Residual Aligner-based Network (RAN): Motion-separable structure for coarse-to-fine discontinuous deformable registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002980)] [[CODE](https://github.com/jianqingzheng/res_aligner_net)] [**`medi.`**]

### ICLR
- FreeReg: Image-to-Point Cloud Registration Leveraging Pretrained Diffusion Models and Monocular Depth Estimators. [[PUB](https://openreview.net/forum?id=BPb5AhT2Vf)]
- A Plug-and-Play Image Registration Network. [[PUB](https://openreview.net/forum?id=DGez4B2a6Y)]

## 2023

### NeurIPS
- Lung250M-4B: A Combined 3D Dataset for CT- and Point Cloud-Based Intra-Patient Lung Registration. [[PUB](http://papers.nips.cc/paper_files/paper/2023/hash/abf37695a4562ac4c05194d717d47eec-Abstract-Datasets_and_Benchmarks.html)] [**`data.`**] [**`pc.`**]
- SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation. [[PUB](https://proceedings.neurips.cc//paper_files/paper/2023/hash/43069caa6776eac8bca4bfd74d4a476d-Abstract-Conference.html)] [**`pc.`**]
- Non-Rigid Shape Registration via Deep Functional Maps Prior. [[PUB](https://proceedings.neurips.cc//paper_files/paper/2023/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html)]
- E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning. [[PUB](https://proceedings.neurips.cc//paper_files/paper/2023/hash/3a2d1bf9bc0a9794cf82c1341a7a75e6-Abstract-Conference.html)] [**`pc.`**]
- Differentiable Registration of Images and LiDAR Point Clouds with VoxelPoint-to-Pixel Matching. [[PUB](https://proceedings.neurips.cc//paper_files/paper/2023/hash/a0a53fefef4c2ad72d5ab79703ba70cb-Abstract-Conference.html)] [**`pc.`**]

### ICCV
- DReg-NeRF: Deep Registration for Neural Radiance Fields. [[PUB](https://ieeexplore.ieee.org/document/10377293)]
- Rethinking Point Cloud Registration as Masking and Reconstruction. [[PUB](https://ieeexplore.ieee.org/document/10377094)] [**`pc.`**]
- SIRA-PCR: Sim-to-Real Adaptation for 3D Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10377707)] [**`pc.`**]
- AutoSynth: Learning to Generate 3D Training Data for Object Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10378635)] [**`pc.`**]
- Preserving Tumor Volumes for Unsupervised Medical Image Registration. [[PUB](https://ieeexplore.ieee.org/document/10377041)]
- Towards Saner Deep Image Registration. [[PUB](https://ieeexplore.ieee.org/document/10377676)]
- Point-TTA: Test-Time Adaptation for Point Cloud Registration Using Multitask Meta-Auxiliary Learning. [[PUB](https://ieeexplore.ieee.org/document/10378313)] [**`pc.`**]
- Chasing clouds: Differentiable volumetric rasterisation of point clouds as a  highly efficient and accurate loss for large-scale deformable 3D registration. [[PUB](https://ieeexplore.ieee.org/document/10378523/)] [**`pc.`**]
- Center-Based Decoupled Point Cloud Registration for 6D Object Pose Estimation. [[PUB](https://ieeexplore.ieee.org/document/10378523/)] [**`pc.`**]
- 2D3D-MATR: 2D-3D Matching Transformer for Detection-free Registration between Images and Point Clouds. [[PUB](https://ieeexplore.ieee.org/document/10378633)] [**`pc.`**]
- RegFormer: An Efficient Projection-Aware Transformer Network for Large-Scale Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10376887)] [**`pc.`**]
- Density-invariant Features for Distant Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10377308)] [**`pc.`**]
- Batch-based Model Registration for Fast 3D Sherd Reconstruction. [[PUB](https://ieeexplore.ieee.org/document/10376539)]
- PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10378216)] [**`pc.`**]
- Occluded Gait Recognition via Silhouette Registration Guided by Automated Occlusion Degree Estimation. [[PUB](https://ieeexplore.ieee.org/document/10350601)]

### CVPR
- BUFFER: Balancing Accuracy, Efficiency, and Generalizability in Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10205493/)] [**`pc.`**]
- Local-to-Global Registration for Bundle-Adjusting Neural Radiance Fields. [[PUB](https://ieeexplore.ieee.org/document/10203801)]
- ObjectMatch: Robust Registration using Canonical Object Correspondences. [[PUB](https://ieeexplore.ieee.org/document/10203222)]
- PEAL: Prior-embedded Explicit Attention Learning for Low-overlap Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10203464)] [**`pc.`**]
- Indescribable Multi-modal Spatial Evaluator. [[PUB](https://arxiv.org/pdf/2303.00369.pdf)] [[CODE](https://github.com/Kid-Liet/IMSE)] [**`medi.`**]
- Robust Outlier Rejection for 3D Registration with Variational Bayes. [[PUB](https://arxiv.org/pdf/2304.01514.pdf)] [[CODE](https://github.com/Jiang-HB/VBReg)] [**`pc.`**]
- 3D Registration with Maximal Cliques. [[PUB](https://arxiv.org/pdf/2305.10854v1.pdf)] [[CODE](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)] [**`pc.`**]
- Deep Graph-based Spatial Consistency for Robust Non-rigid Point Cloud Registration. [[PUB](https://arxiv.org/pdf/2303.09950.pdf)] [[CODE](https://github.com/qinzheng93/GraphSCNet)] [**`pc.`**]
- Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration. [[PUB](https://arxiv.org/pdf/2303.13290v1.pdf)] [**`pc.`**]
- Robust Multiview Point Cloud Registration with Reliable Pose Graph Initialization and History Reweighting. [[PUB](https://arxiv.org/pdf/2304.00467v1.pdf)] [[CODE](https://github.com/WHU-USI3DV/SGHR)] [**`pc.`**]

### AAAI
- Rethinking Rotation Invariance with Point Cloud Registration. [[PUB](https://arxiv.org/pdf/2301.00149v1.pdf)] [[CODE](https://github.com/Crane-YU/rethink_rotation)] [**`pc.`**]
- Fourier-Net: Fast Image Registration with Band-Limited Deformation. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/25182)]
- Stroke Extraction of Chinese Character Based on Deep Structure Deformable Image Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/25220)]

### TPAMI
- RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations. [[PUB](https://ieeexplore.ieee.org/abstract/document/10044259)] [[CODE](https://github.com/HpWang-whu/RoReg)] [**`pc.`**]
- DPCN++: Differentiable Phase Correlation Network for Versatile Pose Registration. [[PUB](https://ieeexplore.ieee.org/document/10256027)]
- SC${2}$2-PCR++: Rethinking the Generation and Selection for Efficient and Robust Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10115040)] [**`pc.`**]
- Robust Point Cloud Registration Framework Based on Deep Graph Matching. [[PUB](https://ieeexplore.ieee.org/document/9878213)] [**`pc.`**]
- Multiway Non-Rigid Point Cloud Registration via Learned Functional Map Synchronization. [[PUB](https://ieeexplore.ieee.org/document/9749887)] [**`pc.`**]
- QGORE: Quadratic-Time Guaranteed Outlier Removal for Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10091912)] [**`pc.`**]
- Sparse-to-Dense Matching Network for Large-Scale LiDAR Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10097640)] [**`pc.`**]
- HRegNet: A Hierarchical Network for Efficient and Accurate Outdoor LiDAR Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/10148795)] [**`pc.`**]
- $\mathcal {X}$-Metric: An N-Dimensional Information-Theoretic Framework for Groupwise Registration and Deep Combined Computing. [[PUB](https://ieeexplore.ieee.org/document/9965747)]
- Learning General and Distinctive 3D Local Deep Descriptors for Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/9775606)] [**`pc.`**]
- GeoTransformer: Fast and Robust Point Cloud Registration With Geometric Transformer. [[PUB](https://ieeexplore.ieee.org/document/10076895)] [**`pc.`**]
- Cycle Registration in Persistent Homology With Applications in Topological Bootstrap. [[PUB](https://ieeexplore.ieee.org/document/9931659)]
- STORM: Structure-Based Overlap Matching for Partial Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/9705149)] [**`pc.`**]
- MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion. [[PUB](https://ieeexplore.ieee.org/document/10145843)]
- A New Outlier Removal Strategy Based on Reliability of Correspondence Graph for Fast Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/9969937)] [**`pc.`**]
- Hunter: Exploring High-Order Consistency for Point Cloud Registration With Severe Outliers. [[PUB](https://ieeexplore.ieee.org/document/10246849)] [**`pc.`**]
- Fast and Robust Non-Rigid Registration Using Accelerated Majorization-Minimization. [[PUB](https://ieeexplore.ieee.org/document/10049724)]

### MICCAI
- A Denoised Mean Teacher for Domain Adaptive Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_63)] [CODE](https://github.com/uncbiag/robot) [**`pc.`**]
- Unsupervised 3D Registration Through Optimization-Guided Cyclical Self-training. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_64)] [CODE](https://github.com/multimodallearning/reg-cyclical-self-train)
- Implicit Neural Representations for Joint Decomposition and Registration of Gene Expression Images in the Marmoset Brain. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_61)] [CODE](https://gene-atlas.brainminds.jp/)
- An Unsupervised Multispectral Image Registration Network for Skin Diseases. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_68)] [CODE](https://github.com/SH-Diao123/MSIR)
- GSMorph: Gradient Surgery for Cine-MRI Cardiac Deformable Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_58)] [CODE](https://github.com/wulalago/GSMorph)
- Inverse Consistency by Construction for Multistep Deep Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_65)] [CODE](https://github.com/uncbiag/ByConstructionICON)
- Learning Expected Appearances for Intraoperative Registration During Neurosurgery. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_22)] [CODE](https://github.com/rouge1616/ExApp/)
- StructuRegNet: Structure-Guided Multimodal 2D-3D Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_73)]
- SAMConvex: Fast Discrete Optimization for CT Registration Using Self-supervised Anatomical Embedding and Correlation Pyramid. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_53)]
- Co-learning Semantic-Aware Unsupervised Segmentation for Pathological Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_51)] [CODE](https://github.com/brain-intelligence-lab/GIRNet)
- PIViT: Large Deformation Image Registration with Pyramid-Iterative Vision Transformer. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_57)] [CODE](https://github.com/Torbjorn1997/PIViT)
- CortexMorph: Fast Cortical Thickness Estimation via Diffeomorphic Registration Using VoxelMorph. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_69)] [CODE](https://github.com/SCAN-NRAD/CortexMorph)
- Non-iterative Coarse-to-Fine Transformer Networks for Joint Affine and Deformable Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_71)]
- FSDiffReg: Feature-Wise and Score-Wise Diffusion-Guided Unsupervised Deformable Image Registration for Cardiac Images. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_62)] [CODE](https://github.com/xmed-lab/FSDiffReg.git)
- WarpEM: Dynamic Time Warping for Accurate Catheter Registration in EM-Guided Procedures. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_75)]
- Regularized Kelvinlet Functions to Model Linear Elasticity for Image-to-Physical Registration of the Breast. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_33)]
- DISA: DIfferentiable Similarity Approximation for Universal Multimodal Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_72)] [CODE](https://github.com/ImFusionGmbH/DISA-universal-multimodal-registration)
- FocalErrorNet: Uncertainty-Aware Focal Modulation Network for Inter-modal Registration Error Estimation in Ultrasound-Guided Neurosurgery. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_66)]
- X-Ray to CT Rigid Registration Using Scene Coordinate Regression. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_74)] [CODE](https://github.com/Pragyanstha/SCR-Registration)
- Nonuniformly Spaced Control Points Based on Variational Cardiac Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_60)]
- Progressively Coupling Network for Brain MRI Registration in Few-Shot Situation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_59)]
- ModeT: Learning Deformable Image Registration via Motion Decomposition Transformer. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_70)] [CODE](https://github.com/ZAX130/SmileCode)
- Importance Weighted Variational Cardiac MRI Registration Using Transformer and Implicit Prior. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_55)]
- TractCloud: Registration-Free Tractography Parcellation with a Novel Local-Global Streamline Point Cloud Representation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_40)]
- A Novel Video-CTU Registration Method with Structural Point Similarity for FURS Navigation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_12)]
- A Patient-Specific Self-supervised Model for Automatic X-Ray/CT Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_49)] [CODE](https://github.com/BaochangZhang/PSSS_registration)
- SPR-Net: Structural Points Based Registration for Coronary Arteries Across Systolic and Diastolic Phases. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-43990-2_74)]

### MIA
- Strain estimation in aortic roots from 4D echocardiographic images using medial modeling and deformable registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000658)]
- Colonoscopy 3D video dataset with paired depth from 2D-3D registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002165)]
- AMNet: Adaptive multi-level network for deformable registration of 3D brain MR images. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000014)]
- DuSFE: Dual-Channel Squeeze-Fusion-Excitation co-attention for cross-modality registration of cardiac SPECT and CT. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001007)]
- Semantic similarity metrics for image registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000907)]
- R2Net: Efficient and flexible diffeomorphic image registration using Lipschitz continuous residual networks. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001779)]
- Anatomically constrained and attention-guided deep feature fusion for joint segmentation and deformable medical image registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000725)]
- Prototypical few-shot segmentation for cross-institution male pelvic structures with spatial registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001950)]
- WarpPINN: Cine-MR image registration with physics-informed neural networks. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523001858)]
- A robust and interpretable deep learning framework for multi-modal registration via keypoints. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002220)]
- PC-Reg: A pyramidal prediction-correction approach for large deformation image registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523002384)]
- DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841522003061)]
- SpineRegNet: Spine Registration Network for volumetric MR and CT image by the joint estimation of an affine-elastic deformation field. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841523000476)]
- QACL: Quartet attention aware closed-loop learning for abdominal MR-to-CT synthesis via simultaneous registration. [[PUB](https://linkinghub.elsevier.com/retrieve/pii/S1361841522003206)]

## 2022

### ECCV
- Bayesian Tracking of Video Graphs Using Joint Kalman Smoothing and Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19833-5_26)]
- Learning-Based Point Cloud Registration for 6D Object Pose Estimation in the Real World. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_2)] [**`pc.`**]
- DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_20)] [[CODE](https://github.com/diffusemorph/diffusemorph)]
- PCR-CG: Point Cloud Registration via Deep Explicit Color and Geometry. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_26)] [[CODE](https://github.com/Gardlin/PCR-CG)]
- Unsupervised Deep Multi-shape Matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_4)] [[CODE](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching)]
- ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_2)] [[CODE](https://github.com/apple/ml-aspanformer)]
- CMT: Context-Matching-Guided Transformer for 3D Tracking in Point Clouds. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_6)] [[CODE](https://github.com/jasongzy/CMT)]
- A Comparative Study of Graph Matching Algorithms in Computer Vision. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_37)]
- Self-supervised Learning of Visual Graph Matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_22)]
- 3DG-STFM: 3D Geometric Guided Student-Teacher Feature Matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19815-1_8)] [[CODE](https://github.com/Ryan-prime/3DG-STFM)]
- Is Geometry Enough for Matching in Visual Localization?. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_24)]
- Explaining Deepfake Detection by Analysing Image Matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19781-9_2)] [[CODE](https://github.com/megvii-research/fst-matching)]
- Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19803-8_35)] [[CODE](https://github.com/ruc-aimc-lab/superretina)]
- DFNet: Enhance Absolute Pose Regression with Direct Feature Matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_1)] [[CODE](https://github.com/activevisionlab/dfnet)]
- Unitail: Detecting, Reading, and Matching in Retail Scene. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_41)]
- Implicit field supervision for robust non-rigid shape matching. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20062-5_20)] [[CODE](https://github.com/Sentient07/IFMatch)]
- Registration based Few-Shot Anomaly Detection. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_18)] [[CODE](https://github.com/mediabrain-sjtu/regad)]
- Improving RGB-D Point Cloud Registration by Learning Multi-scale Local Linear Transformation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-19824-3_11)] [[CODE](https://github.com/514DNA/LLT)] [**`pc.`**]
- PointCLM: A Contrastive Learning-based Framework for Multi-instance Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_35)] [[CODE](https://github.com/phdymz/PointCLM)] [**`pc.`**]
- SuperLine3D: Self-supervised Line Segmentation and Description for LiDAR Point Cloud. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_16)] [[CODE](https://github.com/zxrzju/SuperLine3D)] [**`pc.`**]

### AAAI
- Stochastic Planner-Actor-Critic for Unsupervised Deformable Image Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/20086/19845)] [[CODE](https://github.com/Algolzw/SPAC-Deformable-Registration)] [**`medi.`**]
- DeTarNet: Decoupling Translation and Rotation by Siamese Network for Point Cloud Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/19917)] [[CODE](https://github.com/ZhiChen902/DetarNet)] [**`pc.`**]
- Deep Confidence Guided Distance for 3D Partial Shape Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/19951)]
- Reliable Inlier Evaluation for Unsupervised Point Cloud Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/20117)] [[CODE](https://github.com/supersyq/rienet)] [**`pc.`**]
- FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/20189)] [[CODE](https://github.com/MegEngine/FINet)] [**`pc.`**]
- End-to-End Learning the Partial Permutation Matrix for Robust 3D Point Cloud Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/20250)] [**`pc.`**]

### CVPR
- Deterministic Point Cloud Registration via Novel Transformation Decomposition. [[PUB](https://ieeexplore.ieee.org/document/9878458/)] [**`pc.`**]
- Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment. [[PUB](https://ieeexplore.ieee.org/document/9879348/)] [[CODE](https://github.com/uncbiag/Aladdin)]
- Coherent Point Drift Revisited for Non-rigid Shape Matching and Registration. [[PUB](https://ieeexplore.ieee.org/document/9879560/)] [[CODE](https://github.com/AoxiangFan/GeneralizedCoherentPointDrift)]
- A variational Bayesian method for similarity learning in non-rigid image registration. [[PUB](https://ieeexplore.ieee.org/document/9879941)] [[CODE](https://github.com/dgrzech/learnsim)]
- Affine Medical Image Registration with Coarse-to-Fine Vision Transformer. [[PUB](https://ieeexplore.ieee.org/document/9879546/)] [[CODE](https://github.com/cwmok/C2FViT)]
- Topology-Preserving Shape Reconstruction and Registration via Neural Diffeomorphic Flow. [[PUB](https://ieeexplore.ieee.org/document/9880332/)] [[CODE](https://github.com/Siwensun/Neural_Diffeomorphic_Flow--NDF)]
- Global-Aware Registration of Less-Overlap RGB-D Scans. [[PUB](https://ieeexplore.ieee.org/document/9878484/)] [[CODE](https://github.com/2120171054/Global-Aware-Registration-of-Less-Overlap-RGB-D-Scans)]
- Multi-instance Point Cloud Registration by Efficient Correspondence Clustering. [[PUB](https://ieeexplore.ieee.org/document/9880256/)] [[CODE](https://github.com/Gilgamesh666666/Multi-instance-Point-Cloud-Registration-by-Efficient-Correspondence-Clustering)] [**`pc.`**]
- NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration. [[PUB](https://ieeexplore.ieee.org/document/9879617)] [[CODE](https://github.com/yifannnwu/NODEO-DIR)]
- RFNet: Unsupervised Network for Mutually Reinforcing Multi-modal Image Registration and Fusion. [[PUB](https://ieeexplore.ieee.org/document/9878923)]
- REGTR: End-to-end Point Cloud Correspondences with Transformers. [[PUB](https://arxiv.org/pdf/2203.14517v1.pdf)] [[CODE](https://github.com/yewzijian/RegTR)] [**`pc.`**]
- SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/document/9878510/)] [[CODE](https://github.com/ZhiChen902/SC2-PCR)] [**`pc.`**]
- Geometric Transformer for Fast and Robust Point Cloud Registration. [[PUB](https://arxiv.org/pdf/2202.06688.pdf)] [[CODE](https://github.com/qinzheng93/GeoTransformer)] [**`pc.`**] :fire:
- Lepard: Learning partial point cloud matching in rigid and deformable scenes. [[PUB](https://ieeexplore.ieee.org/document/9878922)] [[CODE](https://github.com/rabbityl/lepard)]

### MICCAI
- Adapting the Mean Teacher for Keypoint-Based Lung Registration Under Geometric Domain Shifts. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_27)]
- Vol2Flow: Segment 3D Volumes Using a Sequence of Registration Flows. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_58)]
- Deformer: Towards Displacement Field Learning for Unsupervised Medical Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_14)]
- Dual-Branch Squeeze-Fusion-Excitation Module for Cross-Modality Registration of Cardiac SPECT and CT. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_5)]
- ContraReg: Contrastive Learning of Multi-modality Unsupervised Deformable Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_7)]
- Learning Iterative Optimisation for Deformable Image Registration of Lung CT with Recurrent Convolutional Networks. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_29)]
- Weakly-Supervised Biomechanically-Constrained CT/MRI Registration of the Spine. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_22)]
- End-to-End Multi-Slice-to-Volume Concurrent Registration and Multimodal Generation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_15)]
- On the Dataset Quality Control for Image Registration Evaluation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_4)]
- DSR: Direct Simultaneous Registration for Multiple 3D Images. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_10)]
- Global Multi-modal 2D/3D Registration via Local Descriptors Learning. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_26)]
- Non-iterative Coarse-to-Fine Registration Based on Single-Pass Deep Cumulative Learning. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_9)] [CODE](https://github.com/MungoMeng/Registration-NICE-Trans)
- An Optimal Control Problem for Elastic Registration and Force Estimation in Augmented Surgery. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_8)]
- Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-recurrence Brain Tumor MRI Scans. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_3)]
- Embedding Gradient-Based Optimization in Image Registration Networks. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_6)]
- Collaborative Quantization Embeddings for Intra-subject Prostate MR Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_23)]
- XMorpher: Full Transformer for Deformable Medical Image Registration via Cross Attention. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_21)]
- Multi-modal Retinal Image Registration Using a Keypoint-Based Vessel Structure Aligning Network. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_11)]
- A Deep-Discrete Learning Framework for Spherical Surface Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_12)]
- Privacy Preserving Image Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_13)]
- LiftReg: Limited Angle 2D/3D Deformable Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_20)]
- Electron Microscope Image Registration Using Laplacian Sharpening Transformer U-Net. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_30)]
- Double-Uncertainty Guided Spatial and Temporal Consistency Regularization Weighting for Learning-Based Abdominal Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_2)]
- SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1)]
- Learning-Based US-MR Liver Image Registration with Spatial Priors. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_17)]
- Swin-VoxelMorph: A Symmetric Unsupervised Learning Model for Deformable Medical Image Registration Using Swin Transformer. [[PUB](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_8)]

### MIA
- Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas. [[PUB](https://doi.org/10.1016/j.media.2021.102265)] [[CODE](https://github.com/acasamitjana/3dhirest)] [**`medi.`**]
- Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance. [[PUB](https://doi.org/10.1016/j.media.2021.102292)] [**`medi.`**]
- Dual-stream pyramid registration network. [[PUB](https://doi.org/10.1016/j.media.2022.102379)] [[Unofficial code](https://github.com/olddriverjinx/reimplemention-of-dual-prnet)] [**`medi.`**]
- Atlas-ISTN: Joint segmentation, registration and atlas construction with image-and-spatial transformer networks. [[PUB](https://doi.org/10.1016/j.media.2022.102383)] [**`medi.`**]

### NeurIPS
- Non-rigid Point Cloud Registration with Neural Deformation Pyramid. [[PUB](https://arxiv.org/pdf/2205.12796.pdf)] [[CODE](https://github.com/rabbityl/DeformationPyramid)]
- One-Inlier is First: Towards Efficient Position Encoding for Point Cloud Registration. [[PUB](https://proceedings.neurips.cc//paper_files/paper/2022/hash/2e163450c1ae3167832971e6da29f38d-Abstract-Conference.html)]

### ICLR
- Partial Wasserstein Adversarial Network for Non-rigid Point Set Registration. [[PUB](https://arxiv.org/pdf/2203.02227.pdf)]

### TIP
- GraphReg: Dynamical Point Cloud Registration with Geometry-aware Graph Signal Processing. [[PUB](https://arxiv.org/pdf/2302.01109.pdf)] [[CODE](https://github.com/zikai1/GraphReg)]

### TPAMI
- Robust Point Cloud Registration Framework Based on Deep Graph Matching. [[PUB](https://ieeexplore.ieee.org/document/9878213)] [[CODE](https://github.com/fukexue/RGM)]
- RIGA: Rotation-Invariant and Globally-Aware Descriptors for Point Cloud Registration. [[PUB](https://arxiv.org/pdf/2209.13252.pdf)]

### SIGGRAPH
- ImLoveNet: Misaligned Image-supported Registration Network for Low-overlap Point Cloud Pairs. [[PUB](https://arxiv.org/pdf/2207.00826.pdf)]

### ACM MM
- You Only Hypothesize Once: Point Cloud Registration with Rotation-equivariant Descriptors. [[PUB](https://arxiv.org/pdf/2109.00182.pdf)] [[CODE](https://github.com/HpWang-whu/YOHO)]

### TVCG
- WSDesc: Weakly Supervised 3D Local Descriptor Learning for Point Cloud Registration. [[PUB](https://arxiv.org/pdf/2108.02740.pdf)] [[CODE](https://github.com/craigleili/WSDesc)]

### RAL
- GraffMatch: Global Matching of 3D Lines and Planes for Wide Baseline LiDAR Registration. [[PUB](https://arxiv.org/pdf/2212.12745.pdf)]

## 2021

### CVPR
- PREDATOR: Registration of 3D Point Clouds with Low Overlap. [[PUB](https://arxiv.org/pdf/2011.13005.pdf)] [[code-pytorch](https://github.com/ShengyuH/OverlapPredator)] [**`pc.`**] :fire:
- SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration. [[PUB](https://github.com/QingyongHu/SpinNet)] [[code-pytorch](https://github.com/QingyongHu/SpinNet)] [**`pc.`**] :fire:
- Robust Point Cloud Registration Framework Based on Deep Graph Matching. [[PUB](https://arxiv.org/abs/2103.04256)] [[CODE](https://github.com/fukexue/RGM)] [**`pc.`**]
- PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency. [[PUB](https://arxiv.org/abs/2103.05465)] [**`pc.`**]
- ReAgent: Point Cloud Registration using Imitation and Reinforcement Learning. [[PUB](https://arxiv.org/abs/2103.15231)] [**`pc.`**]
- DeepI2P: Image-to-Point Cloud Registration via Deep Classification. [[PUB](https://arxiv.org/abs/2104.03501)] [[CODE](https://github.com/lijx10/DeepI2P)] [**`pc.`**] :fire:
- UnsupervisedR&R: Unsupervised Point Cloud Registration via Differentiable Rendering. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/papers/Banani_UnsupervisedRR_Unsupervised_Point_Cloud_Registration_via_Differentiable_Rendering_CVPR_2021_paper.pdf)] [**`pc.`**]
- PointNetLK Revisited. [[PUB](https://arxiv.org/pdf/2008.09527.pdf)] [[CODE](https://github.com/Lilac-Lee/PointNetLK_Revisited)] [**`pc.`**]
- RPSRNet: End-to-End Trainable Rigid Point Set Registration Network Using Barnes-Hut 2D-Tree Representation. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/html/Ali_RPSRNet_End-to-End_Trainable_Rigid_Point_Set_Registration_Network_Using_Barnes-Hut_CVPR_2021_paper.html)]
- Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Camera-Space_Hand_Mesh_Recovery_via_Semantic_Aggregation_and_Adaptive_2D-1D_CVPR_2021_paper.html)]
- Recurrent Multi-View Alignment Network for Unsupervised Surface Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_Recurrent_Multi-View_Alignment_Network_for_Unsupervised_Surface_Registration_CVPR_2021_paper.html)]
- Spatiotemporal Registration for Event-Based Visual Odometry. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Spatiotemporal_Registration_for_Event-Based_Visual_Odometry_CVPR_2021_paper.html)]
- Learning-Based Image Registration With Meta-Regularization. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/html/Safadi_Learning-Based_Image_Registration_With_Meta-Regularization_CVPR_2021_paper.html)]
- Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration. [[PUB](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Locally_Aware_Piecewise_Transformation_Fields_for_3D_Human_Mesh_Registration_CVPR_2021_paper.html)]

### ICCV
- HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration. [[PUB](https://arxiv.org/abs/2107.11992)] [[CODE](https://ispc-group.github.io/hregnet)] [**`pc.`**]
- (Just) A Spoonful of Refinements Helps the Registration Error Go Down. [[PUB](https://arxiv.org/abs/2108.03257)] [**`pc.`**]
- A Robust Loss for Point Cloud Registration. [[PUB](https://arxiv.org/abs/2108.11682)] [**`pc.`**]
- Deep Hough Voting for Robust Global Registration. [[PUB](https://arxiv.org/abs/2109.04310)] [**`pc.`**]
- Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration. [[PUB](https://arxiv.org/abs/2109.06619)] [**`pc.`**]
- LSG-CPD: Coherent Point Drift with Local Surface Geometry for Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_LSG-CPD_Coherent_Point_Drift_With_Local_Surface_Geometry_for_Point_ICCV_2021_paper.pdf)] [[CODE](https://github.com/ChirikjianLab/LSG-CPD)] [**`pc.`**]
- OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf)] [[CODE](https://github.com/megvii-research/OMNet)] [**`pc.`**]
- DeepPRO: Deep Partial Point Cloud Registration of Objects. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_DeepPRO_Deep_Partial_Point_Cloud_Registration_of_Objects_ICCV_2021_paper.pdf)] [**`pc.`**]
- Feature Interactive Representation for Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Feature_Interactive_Representation_for_Point_Cloud_Registration_ICCV_2021_paper.pdf)] [[CODE](https://github.com/Ghostish/BAT)] [**`pc.`**]
- Provably Approximated Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Jubran_Provably_Approximated_Point_Cloud_Registration_ICCV_2021_paper.pdf)] [**`pc.`**]
- Distinctiveness oriented Positional Equilibrium for Point Cloud Registration. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Min_Distinctiveness_Oriented_Positional_Equilibrium_for_Point_Cloud_Registration_ICCV_2021_paper.pdf)] [**`pc.`**]
- PCAM: Product of Cross-Attention Matrices for Rigid Registration of Point Clouds. [[PUB](https://openaccess.thecvf.com/content/ICCV2021/papers/Cao_PCAM_Product_of_Cross-Attention_Matrices_for_Rigid_Registration_of_Point_ICCV_2021_paper.pdf)] [**`pc.`**]
- Generative Adversarial Registration for Improved Conditional Deformable Templates. [[PUB](https://ieeexplore.ieee.org/document/9711216)]
- Deep Hough Voting for Robust Global Registration. [[PUB](https://ieeexplore.ieee.org/document/9710911)]

### AAAI
- Low-Rank Registration Based Manifolds for Convection-Dominated PDEs. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/16116)]
- TAILOR: Teaching with Active and Incremental Learning for Object Registration. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/18031)]

### NeurIPS
- Accurate Point Cloud Registration with Robust Optimal Transport. [[PUB](https://proceedings.neurips.cc/paper/2021/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html)]
- Shape Registration in the Time of Transformers. [[PUB](https://proceedings.neurips.cc/paper/2021/hash/2d3d9d5373f378108cdbd30a3c52bd3e-Abstract.html)]
- CoFiNet: Reliable Coarse-to-fine Correspondences for Robust PointCloud Registration. [[PUB](https://proceedings.neurips.cc/paper/2021/hash/c85b2ea9a678e74fdc8bafe5d0707c31-Abstract.html)]

### Robotics and Autonomous Systems
- A Benchmark for Point Clouds Registration Algorithms. [[PUB](https://www.sciencedirect.com/science/article/abs/pii/S0921889021000191?via%3Dihub)] [[CODE](https://github.com/iralabdisco/point_clouds_registration_benchmark?utm_source=catalyzex.com)] [**`pc.`**]

### TPAMI
- Supervision by Registration and Triangulation for Landmark Detection. [[PUB](https://doi.org/10.1109/TPAMI.2020.2983935)]
- Acceleration of Non-Rigid Point Set Registration With Downsampling and Gaussian Process Regression. [[PUB](https://doi.org/10.1109/TPAMI.2020.3043769)]
- Point Set Registration for 3D Range Scans Using Fuzzy Cluster-Based Metric and Efficient Global Optimization. [[PUB](https://doi.org/10.1109/TPAMI.2020.2978477)]
- Topology-Aware Non-Rigid Point Cloud Registration. [[PUB](https://doi.org/10.1109/TPAMI.2019.2940655)]

### MICCAI
- A Deep Discontinuity-Preserving Image Registration Network. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/013-Paper0479.html)] [CODE](https://github.com/cistib/DDIR) [**`medi.`**]
- A Deep Network for Joint Registration and Parcellation of Cortical Surfaces. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/013-Paper0479.html)] [CODE](https://github.com/zhaofenqiang/JointRegAndParc) [**`medi.`**]
- Conditional Deformable Image Registration with Convolutional Neural Network. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/099-Paper0422.html)] [[CODE](https://github.com/cwmok/Conditional_LapIRN)] [**`medi.`**]
- Cross-modal Attention for MRI and Ultrasound Volume Registration. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/122-Paper0642.html)] [[CODE](https://github.com/DIAL-RPI/Attention-Reg)] [**`medi.`**]
- End-to-end Ultrasound Frame to Volume Registration. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/178-Paper0585.html)] [[CODE](https://github.com/DIAL-RPI/FVR-Net)] [**`medi.`**]
- Learning Unsupervised Parameter-specific Affine Transformation for Medical Images Registration. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/281-Paper0337.html)] [[CODE](https://github.com/xuuuuuuchen/PASTA)] [**`medi.`**]
- Multi-view analysis of unregistered medical images using cross-view transformers. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/337-Paper1007.html)] [[CODE](https://github.com/gvtulder/cross-view-transformers)] [**`medi.`**]
- Revisiting iterative highly efficient optimisation schemes in medical image registration. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/405-Paper2461.html)] [[CODE](https://github.com/multimodallearning/iter_lbp)] [**`medi.`**]
- Unsupervised Diffeomorphic Surface Registration and Non-Linear Modelling. [[PUB](https://miccai2021.org/openaccess/paperlinks/2021/09/01/512-Paper0947.html)] [[CODE](https://gitlab.kuleuven.be/u0132345/deepdiffeomorphicfaceregistration)] [**`medi.`**]

### MIA
- A hybrid, image-based and biomechanics-based registration approach to markerless intraoperative nodule localization during video-assisted thoracoscopic surgery. [[PUB](https://doi.org/10.1016/j.media.2021.101983)] [**`medi.`**]
- Real-time multimodal image registration with partial intraoperative point-set data. [[PUB](https://doi.org/10.1016/j.media.2021.102231)] [**`medi.`**]
- Leveraging unsupervised image registration for discovery of landmark shape descriptor. [[PUB](https://doi.org/10.1016/j.media.2021.102157)] [**`medi.`**]
- Weakly-supervised learning of multi-modal features for regularised iterative descent in 3D image registration. [[PUB](https://doi.org/10.1016/j.media.2020.101822)] [**`medi.`**]
- Shape registration with learned deformations for 3D shape reconstruction from sparse and incomplete point clouds. [[PUB](https://doi.org/10.1016/j.media.2021.102228)] [**`medi.`**]
- Variational multi-task MRI reconstruction: Joint reconstruction, registration and super-resolution. [[PUB](https://doi.org/10.1016/j.media.2020.101941)] [**`medi.`**]
- A novel approach to 2D/3D registration of X-ray images using Grangeat's relation. [[PUB](https://doi.org/10.1016/j.media.2020.101815)] [**`medi.`**]
- Biomechanically constrained non-rigid MR-TRUS prostate registration using deep learning based 3D point cloud matching. [[PUB](https://doi.org/10.1016/j.media.2020.101845)] [**`medi.`**]
- Fracture reduction planning and guidance in orthopaedic trauma surgery via multi-body image registration. [[PUB](https://doi.org/10.1016/j.media.2020.101917)] [**`medi.`**]
- CNN-based lung CT registration with multiple anatomical constraints. [[PUB](https://doi.org/10.1016/j.media.2021.102139)] [**`medi.`**]
- End-to-end multimodal image registration via reinforcement learning. [[PUB](https://doi.org/10.1016/j.media.2020.101878)] [**`medi.`**]
- Difficulty-aware hierarchical convolutional neural networks for deformable registration of brain MR images. [[PUB](https://doi.org/10.1016/j.media.2020.101817)] [**`medi.`**]
- CycleMorph: Cycle consistent unsupervised deformable image registration. [[PUB](https://doi.org/10.1016/j.media.2021.102036)] [**`medi.`**]
- Rethinking medical image reconstruction via shape prior, going deeper and faster: Deep joint indirect registration and reconstruction. [[PUB](https://doi.org/10.1016/j.media.2020.101930)] [**`medi.`**]
- Deformation analysis of surface and bronchial structures in intraoperative pneumothorax using deformable mesh registration. [[PUB](https://doi.org/10.1016/j.media.2021.102181)] [**`medi.`**]
- Re-Identification and growth detection of pulmonary nodules without image registration using 3D siamese neural networks. [[PUB](https://doi.org/10.1016/j.media.2020.101823)] [**`medi.`**]
- Image registration: Maximum likelihood, minimum entropy and deep learning. [[PUB](https://doi.org/10.1016/j.media.2020.101939)] [**`medi.`**]
- ProsRegNet: A deep learning framework for registration of MRI and histopathology images of the prostate. [[PUB](https://doi.org/10.1016/j.media.2020.101919)] [**`medi.`**]
- 3D Registration of pre-surgical prostate MRI and histopathology images via super-resolution volume reconstruction. [[PUB](https://doi.org/10.1016/j.media.2021.101957)] [**`medi.`**]
- A deep learning framework for pancreas segmentation with multi-atlas registration and 3D level-set. [[PUB](https://doi.org/10.1016/j.media.2020.101884)] [**`medi.`**]
- Anatomy-guided multimodal registration by learning segmentation without ground truth: Application to intraprocedural CBCT/MR liver segmentation and registration. [[PUB](https://doi.org/10.1016/j.media.2021.102041)] [**`medi.`**]

## 2020

### CVPR
- Learning multiview 3D point cloud registration. [[PUB](https://arxiv.org/abs/2001.05119)] [[CODE](https://github.com/zgojcic/3D_multiview_reg)] [**`pc.`**] :fire:
- SampleNet: Differentiable Point Cloud Sampling. [[PUB](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf)] [[CODE](https://github.com/itailang/SampleNet)] [**`pc.`**] :fire:
- Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences. [[PUB](https://arxiv.org/abs/2005.01014)] [[CODE](https://github.com/XiaoshuiHuang/fmr)] [**`pc.`**]
- Deep Global Registration. [[PUB](https://arxiv.org/abs/2004.11540)] [**`pc.`**]
- Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Arar_Unsupervised_Multi-Modal_Image_Registration_via_Geometry_Preserving_Image-to-Image_Translation_CVPR_2020_paper.html)]
- Smooth Shells: Multi-Scale Shape Registration With Functional Maps. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Eisenberger_Smooth_Shells_Multi-Scale_Shape_Registration_With_Functional_Maps_CVPR_2020_paper.html)]
- Global Optimality for Point Set Registration Using Semidefinite Programming. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Iglesias_Global_Optimality_for_Point_Set_Registration_Using_Semidefinite_Programming_CVPR_2020_paper.html)]
- Fast Symmetric Diffeomorphic Image Registration with Convolutional Neural Networks. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Mok_Fast_Symmetric_Diffeomorphic_Image_Registration_with_Convolutional_Neural_Networks_CVPR_2020_paper.html)]
- 3DRegNet: A Deep Neural Network for 3D Point Registration. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Pais_3DRegNet_A_Deep_Neural_Network_for_3D_Point_Registration_CVPR_2020_paper.html)]
- DeepFLASH: An Efficient Network for Learning-Based Medical Image Registration. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_DeepFLASH_An_Efficient_Network_for_Learning-Based_Medical_Image_Registration_CVPR_2020_paper.html)]
- Quasi-Newton Solver for Robust Non-Rigid Registration. [[PUB](https://openaccess.thecvf.com/content_CVPR_2020/html/Yao_Quasi-Newton_Solver_for_Robust_Non-Rigid_Registration_CVPR_2020_paper.html)]

### NeurIPS
- LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration. [[PUB](https://proceedings.neurips.cc/paper/2020/hash/970af30e481057c48f87e101b61e6994-Abstract.html)]
- CoMIR: Contrastive Multimodal Image Representation for Registration. [[PUB](https://proceedings.neurips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html)]

### ECCV
- Deep Complementary Joint Model for Complex Scene Registration and Few-Shot Segmentation on Medical Images. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-58523-5_45)]
- Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_23)]
- JSSR: A Joint Synthesis, Segmentation, and Registration System for 3D Multi-modal Image Alignment of Large-Scale Pathological CT Scans. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_16)]
- A Closest Point Proposal for MCMC-based Probabilistic Surface Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_17)]
- DeepGMR: Learning Latent Gaussian Mixture Models for Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_43)]

### 3DV
- Registration Loss Learning for Deep Probabilistic Point Set Registration. [[PUB](https://arxiv.org/abs/2011.02229)] [[code-pytorch](https://github.com/felja633/RLLReg)] [**`pc.`**]

### TPAMI
- Aggregated Wasserstein Distance and State Registration for Hidden Markov Models. [[PUB](https://doi.org/10.1109/TPAMI.2019.2908635)]

### MICCAI
- MvMM-RegNet: A New Image Registration Framework Based on Multivariate Mixture Model and Neural Network Estimation. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_15)] [[CODE](https://zmiclab.github.io/projects.html)] [**`medi.`**]
- Highly Accurate and Memory Efficient Unsupervised Learning-Based Discrete CT Registration Using 2.5D Displacement Search. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_19)] [[CODE](https://github.com/multimodallearning/pdd2.5/)] [**`medi.`**]
- Generalizing Spatial Transformers to Projective Geometry with Applications to 2D/3D Registration. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_32)] [[CODE](https://github.com/gaocong13/Projective-Spatial-Transformers)] [**`medi.`**]
- Non-Rigid Volume to Surface Registration Using a Data-Driven Biomechanical Model. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-59719-1_70)] [[CODE](https://gitlab.com/nct_tso_public/Volume2SurfaceCNN)] [**`medi.`**]

### MIA
- Hubless keypoint-based 3D deformable groupwise registration. [[PUB](https://doi.org/10.1016/j.media.2019.101564)] [**`medi.`**]
- Multi-atlas image registration of clinical data with automated quality assessment using ventricle segmentation. [[PUB](https://doi.org/10.1016/j.media.2020.101698)] [**`medi.`**]
- Groupwise registration with global-local graph shrinkage in atlas construction. [[PUB](https://doi.org/10.1016/j.media.2020.101711)] [**`medi.`**]
- SLIR: Synthesis, localization, inpainting, and registration for image-guided thermal ablation of liver tumors. [[PUB](https://doi.org/10.1016/j.media.2020.101763)] [**`medi.`**]

## 2019

### CVPR
- DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds. [[PUB](https://arxiv.org/abs/1811.11397)] [[CODE](https://ai4ce.github.io/DeepMapping/)] [**`pc.`**] :fire:
- PointNetLK: Point Cloud Registration using PointNet. [[PUB](https://arxiv.org/abs/1903.05711)] [[code-pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**] :fire:
- SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. [[PUB](https://arxiv.org/abs/1904.03483)] [[matlab](https://github.com/intellhave/SDRSAC)] [**`pc.`**]
- FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. [[PUB](https://arxiv.org/abs/1811.10136)] [[CODE](https://bitbucket.org/gaowei19951004/poser/src/master/)] [**`pc.`**]
- PointNetLK: Robust & Efficient Point Cloud Registration using PointNet. [[PUB](https://arxiv.org/abs/1903.05711)] [[code-pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**] :fire:
- 3D Local Features for Direct Pairwise Registration. [[PUB](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] [**`pc.`**]
- Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation. [[PUB](http://openaccess.thecvf.com/content_CVPR_2019/html/Liao_Multiview_2D3D_Rigid_Registration_via_a_Point-Of-Interest_Network_for_Tracking_CVPR_2019_paper.html)]
- Metric Learning for Image Registration. [[PUB](http://openaccess.thecvf.com/content_CVPR_2019/html/Niethammer_Metric_Learning_for_Image_Registration_CVPR_2019_paper.html)]
- Networks for Joint Affine and Non-Parametric Image Registration. [[PUB](http://openaccess.thecvf.com/content_CVPR_2019/html/Shen_Networks_for_Joint_Affine_and_Non-Parametric_Image_Registration_CVPR_2019_paper.html)]

### ICCV
- DeepVCP: An End-to-End Deep Neural Network for 3D Point Cloud Registration. [[PUB](https://arxiv.org/abs/1905.04153v2)] [**`pc.`**]
- Accelerated Gravitational Point Set Alignment with Altered Physical Laws. [[PUB](http://openaccess.thecvf.com/content_ICCV_2019/papers/Golyanik_Accelerated_Gravitational_Point_Set_Alignment_With_Altered_Physical_Laws_ICCV_2019_paper.pdf)] [**`pc.`**]
- Deep Closest Point: Learning Representations for Point Cloud Registration. [[PUB](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)] [**`pc.`**]
- Efficient Learning on Point Clouds with Basis Point Sets. [[PUB](http://openaccess.thecvf.com/content_ICCV_2019/papers/Prokudin_Efficient_Learning_on_Point_Clouds_With_Basis_Point_Sets_ICCV_2019_paper.pdf)] [[CODE](https://github.com/sergeyprokudin/bps)] [ **`pc.`**] :fire:
- Robust Variational Bayesian Point Set Registration. [[PUB](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Robust_Variational_Bayesian_Point_Set_Registration_ICCV_2019_paper.pdf)] [**`pc.`**]
- Efficient and Robust Registration on the 3D Special Euclidean Group. [[PUB](https://ieeexplore.ieee.org/document/9008111)]
- Linearly Converging Quasi Branch and Bound Algorithms for Global Rigid Registration. [[PUB](https://ieeexplore.ieee.org/document/9010695)]
- A Deep Step Pattern Representation for Multimodal Retinal Image Registration. [[PUB](https://ieeexplore.ieee.org/document/9008309)]
- Recursive Cascaded Networks for Unsupervised Medical Image Registration. [[PUB](https://ieeexplore.ieee.org/document/9010680)]
- Automatic and Robust Skull Registration Based on Discrete Uniformization. [[PUB](https://ieeexplore.ieee.org/document/9008291)]

### NeurIPS
- Arbicon-Net: Arbitrary Continuous Geometric Transformation Networks for Image Registration. [[PUB](https://proceedings.neurips.cc/paper/2019/hash/56f9f88906aebf4ad985aaec7fa01313-Abstract.html)]
- Recurrent Registration Neural Networks for Deformable Image Registration. [[PUB](https://proceedings.neurips.cc/paper/2019/hash/dd03de08bfdff4d8ab01117276564cc7-Abstract.html)]
- PRNet: Self-Supervised Learning for Partial-to-Partial Registration. [[PUB](https://proceedings.neurips.cc/paper/2019/hash/ebad33b3c9fa1d10327bb55f9e79e2f3-Abstract.html)]

### TPAMI
- Efficient Registration of High-Resolution Feature Enhanced Point Clouds. [[PUB](https://doi.org/10.1109/TPAMI.2018.2831670)]

### MICCAI
- Closing the Gap between Deep and Conventional Image Registration using Probabilistic Dense Displacement Networks. [[PUB](https://arxiv.org/abs/1907.10931)] [[CODE](https://github.com/multimodallearning/pdd_net)] [**`medi.`**]

### ICRA
- Robust low-overlap 3-D point cloud registration for outlier rejection. [[PUB](https://ieeexplore.ieee.org/abstract/document/8793857)] [[matlab](https://github.com/JStech/ICP)] [**`pc.`**]
- Robust Generalized Point Set Registration Using Inhomogeneous Hybrid Mixture Models Via Expectation. [[PUB](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_3.html)] [**`pc.`**]
- CELLO-3D: Estimating the Covariance of ICP in the Real World. [[PUB](https://export.arxiv.org/abs/1810.01470)] [**`pc.`**]

## 2018

### CVPR
- Density Adaptive Point Set Registration. [[PUB](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf)] [[CODE](https://github.com/felja633/DARE)] [**`pc.`**]
- Inverse Composition Discriminative Optimization for Point Cloud Registration. [[PUB](http://openaccess.thecvf.com/content_cvpr_2018/papers/Vongkulbhisal_Inverse_Composition_Discriminative_CVPR_2018_paper.pdf)] [**`pc.`**]
- An Unsupervised Learning Model for Deformable Medical Image Registration. [[PUB](http://openaccess.thecvf.com/content_cvpr_2018/html/Balakrishnan_An_Unsupervised_Learning_CVPR_2018_paper.html)]
- Supervision-by-Registration: An Unsupervised Approach to Improve the Precision of Facial Landmark Detectors. [[PUB](http://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Supervision-by-Registration_An_Unsupervised_CVPR_2018_paper.html)]
- CNN Driven Sparse Multi-Level B-Spline Image Registration. [[PUB](http://openaccess.thecvf.com/content_cvpr_2018/html/Jiang_CNN_Driven_Sparse_CVPR_2018_paper.html)]
- 3D Registration of Curves and Surfaces Using Local Differential Information. [[PUB](http://openaccess.thecvf.com/content_cvpr_2018/html/Raposo_3D_Registration_of_CVPR_2018_paper.html)]

### AAAI
- Dilated FCN for Multi-Agent 2D/3D Medical Image Registration. [[PUB](https://web.archive.org/web/*/https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16085)]

### ECCV
- Learning and Matching Multi-View Descriptors for Registration of Point Clouds. [[PUB](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lei_Zhou_Learning_and_Matching_ECCV_2018_paper.pdf)] [**`pc.`**]
- 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. [[PUB](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] [[code-tensorflow](https://github.com/yewzijian/3DFeatNet)] [**`pc.`**] :fire:
- Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search. [[PUB](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf)] [**`pc.`**]
- HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration. [[PUB](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)] [**`pc.`**]
- Robust Image Stitching with Multiple Registrations. [[PUB](https://link.springer.com/chapter/10.1007/978-3-030-01216-8_4)]

### TPAMI
- Guaranteed Outlier Removal for Point Cloud Registration with Correspondences. [[PUB](https://doi.org/10.1109/TPAMI.2017.2773482)] [**`pc.`**]
- Collocation for Diffeomorphic Deformations in Medical Image Registration. [[PUB](https://doi.org/10.1109/TPAMI.2017.2730205)] [**`medi.`**]
- Hierarchical Sparse Representation for Robust Image Registration. [[PUB](https://doi.org/10.1109/TPAMI.2017.2748125)]
- Multiresolution Search of the Rigid Motion Space for Intensity-Based Registration. [[PUB](https://doi.org/10.1109/TPAMI.2017.2654245)]

### 3DV
- PCN: Point Completion Network. [[PUB](https://arxiv.org/abs/1808.00671)] [[code-tensorflow](https://github.com/TonythePlaneswalker/pcn)] [**`pc.`**] :fire:

### ICRA
- Robust Generalized Point Cloud Registration Using Hybrid Mixture Model. [[PUB](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460825)] [**`pc.`**]
- A General Framework for Flexible Multi-Cue Photometric Point Cloud Registration. [[PUB](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461049)] [**`pc.`**]

### IROS
- Dynamic Scaling Factors of Covariances for Accurate 3D Normal Distributions Transform Registration. [[PUB](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593839)] [**`pc.`**]
- Robust Generalized Point Cloud Registration with Expectation Maximization Considering Anisotropic Positional Uncertainties. [[PUB](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593558)] [**`pc.`**]
- PCAOT: A Manhattan Point Cloud Registration Method Towards Large Rotation and Small Overlap. [[PUB](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594514)] [**`pc.`**]

### IEEE Access
- Multi-temporal Remote Sensing Image Registration Using Deep Convolutional Features. [[PUB](https://ieeexplore.ieee.org/document/8404075)] [[CODE](https://github.com/yzhq97/cnn-registration)] [**`rs.`**] :fire:

## 2017

### CVPR
- 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions. [[PUB](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)] [[CODE](https://github.com/andyzeng/3dmatch-toolbox)] [**`pc.`**] [**`data.`**] :fire: :star:
- Discriminative Optimization: Theory and Applications to Point Cloud Registration. [[PUB](http://openaccess.thecvf.com/content_cvpr_2017/papers/Vongkulbhisal_Discriminative_Optimization_Theory_CVPR_2017_paper.pdf)] [**`pc.`**]
- 3D Point Cloud Registration for Localization using a Deep Neural Network Auto-Encoder. [[PUB](http://openaccess.thecvf.com/content_cvpr_2017/papers/Elbaz_3D_Point_Cloud_CVPR_2017_paper.pdf)] [[CODE](https://github.com/gilbaz/LORAX)] [**`pc.`**]
- Convex Global 3D Registration with Lagrangian Duality. [[PUB](https://ieeexplore.ieee.org/document/8100078)]
- Group-Wise Point-Set Registration Based on Rényi's Second Order Entropy. [[PUB](https://ieeexplore.ieee.org/document/8099746)]
- Fine-to-Coarse Global Registration of RGB-D Scans. [[PUB](https://ieeexplore.ieee.org/document/8100188)]
- Joint Registration and Representation Learning for Unconstrained Face Identification. [[PUB](https://ieeexplore.ieee.org/document/8099652)]
- A General Framework for Curve and Surface Comparison and Registration with Oriented Varifolds. [[PUB](https://ieeexplore.ieee.org/document/8099970)]

### ICCV
- Colored Point Cloud Registration Revisited. [[PUB](http://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)] [**`pc.`**]
- Local-to-Global Point Cloud Registration Using a Dictionary of Viewpoint Descriptors. [[PUB](https://ieeexplore.ieee.org/document/8237364)]
- Joint Layout Estimation and Global Multi-view Registration for Indoor Reconstruction. [[PUB](https://ieeexplore.ieee.org/document/8237289)]
- Deep Free-Form Deformation Network for Object-Mask Registration. [[PUB](https://ieeexplore.ieee.org/document/8237718)]
- Point Set Registration with Global-Local Correspondence and Transformation Estimation. [[PUB](https://ieeexplore.ieee.org/document/8237553)]
- Surface Registration via Foliation. [[PUB](https://ieeexplore.ieee.org/document/8237369)]

### AAAI
- An Artificial Agent for Robust Image Registration. [[PUB](https://arxiv.org/pdf/1611.10336.pdf)]
- Non-Rigid Point Set Registration with Robust Transformation Estimation under Manifold Regularization. [[PUB](https://ojs.aaai.org/index.php/AAAI/article/view/11195)]

### ICRA
- Using 2 point+normal sets for fast registration of point clouds with small overlap. [[PUB](https://ieeexplore.ieee.org/document/7989664)] [**`pc.`**]

### TPAMI
- Image Registration and Change Detection under Rolling Shutter Motion Blur. [[PUB](https://doi.org/10.1109/TPAMI.2016.2630687)]
- Hyperbolic Harmonic Mapping for Surface Registration. [[PUB](https://doi.org/10.1109/TPAMI.2016.2567398)]
- Randomly Perturbed B-Splines for Nonrigid Image Registration. [[PUB](https://doi.org/10.1109/TPAMI.2016.2598344)]

## 2016

### ECCV
- Fast Global Registration. [[PUB](https://www.researchgate.net/profile/Vladlen_Koltun/publication/305983982_Fast_Global_Registration/links/57a8086908aefe6167bc8366/Fast-Global-Registration.pdf)] [CODE](https://github.com/intel-isl/FastGlobalRegistration)

## 2015

### TPAMI
- Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration. [[PUB](https://arxiv.org/pdf/1605.03344.pdf)] [CODE](https://github.com/yangjiaolong/Go-ICP)

## 2009

### ICRA
- Fast point feature histograms (FPFH) for 3D registration. [[PUB](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf)]

### RSS
- Generalized-ICP. [[PUB](http://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf)]

## 1992

### TPAMI
- A method for registration of 3-D shapes. [[PUB](https://www.researchgate.net/publication/3191994_A_method_for_registration_of_3-D_shapes_IEEE_Trans_Pattern_Anal_Mach_Intell)]

## 1987

### TPAMI
- Least-squares fitting of two 3-D point sets. [[PUB](https://www.researchgate.net/publication/224378053_Least-squares_fitting_of_two_3-D_point_sets_IEEE_T_Pattern_Anal)]
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
- [**Zi LI**](https://github.com/Alison-brie) [Awesome-Medical-Image-Registration](https://github.com/Alison-brie/Awesome-Medical-Image-Registration)
