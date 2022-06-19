# image-registration-resources

image registration related books, papers, videos, and toolboxes 

[![Stars](https://img.shields.io/github/stars/youngfish42/image-registration-resources.svg?color=orange)](https://github.com/youngfish42/image-registration-resources/stargazers)  [![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E5%9B%BE%E5%83%8F%E9%85%8D%E5%87%86%E6%8C%87%E5%8C%97-blue)](https://zhuanlan.zhihu.com/Image-Registration)  [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re) [![License](https://img.shields.io/github/license/youngfish42/image-registration-resources.svg?color=green)](https://github.com/youngfish42/image-registration-resources/blob/master/LICENSE) 

[**Image registration**](https://en.wikipedia.org/wiki/Image_registration) is the process of transforming different sets of data into one coordinate system. Data may be multiple photographs, and from different sensors, times, depths, or viewpoints.

It is used in computer vision, medical imaging, military automatic target recognition, compiling and analyzing images and data from satellites. Registration is necessary in order to be able to compare or integrate the data obtained from different measurements. 



[toc]

---

# Paper Lists

A paper list of image registration. 

###  Keywords 

 **`medi.`**: medical image |  **`nat.`**: natural image |  **`rs.`**: remote sensing   |  **`pc.`**: point cloud

 **`data.`**: dataset  |   **`dep.`**: deep learning

 **`oth.`**: other, including  correspondence, mapping, matching, alignment...

Statistics: 🔥 code is available & stars >= 100  |  ⭐ citation >= 50



### Update log

*Last updated: 2022/06/19*

*2022/06/19* - update recent TPAMI papers (2017-2021) about image registration according to dblp search engine, update recent MICCAI papers (2019-2021) about image registration according to [JunMa11](https://github.com/JunMa11)'s [MICCAI-OpenSourcePapers](https://github.com/JunMa11/MICCAI-OpenSourcePapers).

- [TPAMI](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AMedical_Image_Anal.%3A) 
- [MICCAI](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AMedical_Image_Anal.%3A) 

*2022/06/18* - update recent papers (2017-2021) on CVPR/ICCV/ECCV/AAAI/NeurIPS/MIA about image registration according to dblp search engine.

- [CVPR](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3ACVPR%3A) / [ICCV](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3AICCV%3A) / [ECCV](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3AECCV%3A)
- [AAAI](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3AAAAI%3A) / [NeurIPS](https://dblp.org/search?q=registra%20type%3AConference_and_Workshop_Papers%3A%20venue%3ANeurIPS%3A) 
- [MIA](https://dblp.org/search?q=registra%20type%3AJournal_Articles%3A%20venue%3AMedical_Image_Anal.%3A) 

*2022/06/18* - update papers (2020-2022) about point cloud registration from [NUAAXQ](https://github.com/NUAAXQ)'s [awesome-point-cloud-analysis-2022](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2022).

*2020/04/20* - update recent papers (2017-2020) about point cloud registration and make some diagram about history of image registration.



## 2022

[[CVPR](https://arxiv.org/pdf/2203.14517v1.pdf)] REGTR: End-to-end Point Cloud Correspondences with Transformers. [[code](https://github.com/yewzijian/RegTR)] [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2203.14453v1.pdf)] SC2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration. [[code](https://github.com/ZhiChen902/SC2-PCR)] [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2202.06688.pdf)] Geometric Transformer for Fast and Robust Point Cloud Registration. [[code](https://github.com/qinzheng93/GeoTransformer)] [**`pc.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102265)] Robust joint registration of multiple stains and MRI for multimodal 3D histology reconstruction: Application to the Allen human brain atlas. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2021.102292)] Deformable MR-CT image registration using an unsupervised, dual-channel network for neurosurgical guidance. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2022.102379)] Dual-stream pyramid registration network. [**`medi.`**]

[[MIA](https://doi.org/10.1016/j.media.2022.102383)] Atlas-ISTN: Joint segmentation, registration and atlas construction with image-and-spatial transformer networks. [**`medi.`**]



## 2021

[[CVPR](https://arxiv.org/pdf/2011.13005.pdf)] PREDATOR: Registration of 3D Point Clouds with Low Overlap. [[pytorch](https://github.com/ShengyuH/OverlapPredator)] [**`pc.`**]

[[CVPR](https://github.com/QingyongHu/SpinNet)] SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration. [[pytorch](https://github.com/QingyongHu/SpinNet)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/2103.04256)] Robust Point Cloud Registration Framework Based on Deep Graph Matching. [[code](https://github.com/fukexue/RGM)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/2103.05465)] PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency. [**`pc.`**]

[[CVPR](https://arxiv.org/abs/2103.15231)] ReAgent: Point Cloud Registration using Imitation and Reinforcement Learning.[**`pc.`**]

[[CVPR](https://arxiv.org/abs/2104.03501)] DeepI2P: Image-to-Point Cloud Registration via Deep Classification. [[code](https://github.com/lijx10/DeepI2P)] [**`pc.`**]

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/papers/Banani_UnsupervisedRR_Unsupervised_Point_Cloud_Registration_via_Differentiable_Rendering_CVPR_2021_paper.pdf)] UnsupervisedR&R: Unsupervised Point Cloud Registration via Differentiable Rendering. [**`pc.`**]

[[CVPR](https://arxiv.org/pdf/2008.09527.pdf)] PointNetLK Revisited. [[code](https://github.com/Lilac-Lee/PointNetLK_Revisited)] [**`pc.`**]

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Ali_RPSRNet_End-to-End_Trainable_Rigid_Point_Set_Registration_Network_Using_Barnes-Hut_CVPR_2021_paper.html)] RPSRNet: End-to-End Trainable Rigid Point Set Registration Network Using Barnes-Hut 2D-Tree Representation

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Camera-Space_Hand_Mesh_Recovery_via_Semantic_Aggregation_and_Adaptive_2D-1D_CVPR_2021_paper.html)] Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Feng_Recurrent_Multi-View_Alignment_Network_for_Unsupervised_Surface_Registration_CVPR_2021_paper.html)] Recurrent Multi-View Alignment Network for Unsupervised Surface Registration.

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Spatiotemporal_Registration_for_Event-Based_Visual_Odometry_CVPR_2021_paper.html)] Spatiotemporal Registration for Event-Based Visual Odometry.

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Safadi_Learning-Based_Image_Registration_With_Meta-Regularization_CVPR_2021_paper.html)] Learning-Based Image Registration With Meta-Regularization. 

[[CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Locally_Aware_Piecewise_Transformation_Fields_for_3D_Human_Mesh_Registration_CVPR_2021_paper.html)] Locally Aware Piecewise Transformation Fields for 3D Human Mesh Registration.

[[ICCV](https://arxiv.org/abs/2107.11992)] HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration. [[code](https://ispc-group.github.io/hregnet?utm_source=catalyzex.com)] [**`pc.`**]

[[ICCV oral](https://arxiv.org/abs/2108.03257)] (Just) A Spoonful of Refinements Helps the Registration Error Go Down. [**`pc.`**]

[[ICCV](https://arxiv.org/abs/2108.11682)] A Robust Loss for Point Cloud Registration. [**`pc.`**]

[[ICCV](https://arxiv.org/abs/2109.04310)] Deep Hough Voting for Robust Global Registration. [**`pc.`**]

[[ICCV](https://arxiv.org/abs/2109.06619)] Sampling Network Guided Cross-Entropy Method for Unsupervised Point Cloud Registration.[**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_LSG-CPD_Coherent_Point_Drift_With_Local_Surface_Geometry_for_Point_ICCV_2021_paper.pdf)] LSG-CPD: Coherent Point Drift with Local Surface Geometry for Point Cloud Registration.[[code](https://github.com/ChirikjianLab/LSG-CPD)] [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_OMNet_Learning_Overlapping_Mask_for_Partial-to-Partial_Point_Cloud_Registration_ICCV_2021_paper.pdf)] OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration.[[code](https://github.com/megvii-research/OMNet)] [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_DeepPRO_Deep_Partial_Point_Cloud_Registration_of_Objects_ICCV_2021_paper.pdf)] DeepPRO: Deep Partial Point Cloud Registration of Objects. [**`pc.`**]

[[ICCV](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Feature_Interactive_Representation_for_Point_Cloud_Registration_ICCV_2021_paper.pdf)] Feature Interactive Representation for Point Cloud Registration.[[code](https://github.com/Ghostish/BAT)] [**`pc.`**]

[[ICCV](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2022/blob/master/openaccess.thecvf.com/content/ICCV2021/papers/Jubran_Provably_Approximated_Point_Cloud_Registration_ICCV_2021_paper.pdf)] Provably Approximated Point Cloud Registration. [**`pc.`**]

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

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/013-Paper0479.html)] A Deep Discontinuity-Preserving Image Registration Network. [**`medi.`**]

[[MICCAI](https://miccai2021.org/openaccess/paperlinks/2021/09/01/013-Paper0479.html)] A Deep Network for Joint Registration and Parcellation of Cortical Surfaces. [**`medi.`**]

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

[[CVPR](https://arxiv.org/abs/2001.05119)] Learning multiview 3D point cloud registration. [[code](https://github.com/zgojcic/3D_multiview_reg)] [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Lang_SampleNet_Differentiable_Point_Cloud_Sampling_CVPR_2020_paper.pdf)] SampleNet: Differentiable Point Cloud Sampling. [[code](https://github.com/itailang/SampleNet)] [**`pc.`**]

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

[[3DV](https://arxiv.org/abs/2011.02229)] Registration Loss Learning for Deep Probabilistic Point Set Registration. [[pytorch](https://github.com/felja633/RLLReg)] [**`pc.`**]

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

[[CVPR](https://arxiv.org/abs/1811.11397)] DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds. [[code](https://ai4ce.github.io/DeepMapping/)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Point Cloud Registration using PointNet. [[pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1904.03483)] SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. [[matlab](https://github.com/intellhave/SDRSAC)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1811.10136)] FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. [[code](https://bitbucket.org/gaowei19951004/poser/src/master/)] [**`pc.`**]

[[CVPR](https://arxiv.org/abs/1903.05711)] PointNetLK: Robust & Efficient Point Cloud Registration using PointNet. [[pytorch](https://github.com/hmgoforth/PointNetLK)] [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_3D_Local_Features_for_Direct_Pairwise_Registration_CVPR_2019_paper.pdf)] 3D Local Features for Direct Pairwise Registration. [**`pc.`**]

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/html/Liao_Multiview_2D3D_Rigid_Registration_via_a_Point-Of-Interest_Network_for_Tracking_CVPR_2019_paper.html)] Multiview 2D/3D Rigid Registration via a Point-Of-Interest Network for Tracking and Triangulation.

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/html/Niethammer_Metric_Learning_for_Image_Registration_CVPR_2019_paper.html)] Metric Learning for Image Registration.

[[CVPR](http://openaccess.thecvf.com/content_CVPR_2019/html/Shen_Networks_for_Joint_Affine_and_Non-Parametric_Image_Registration_CVPR_2019_paper.html)] Networks for Joint Affine and Non-Parametric Image Registration.

[[ICCV](https://arxiv.org/abs/1905.04153v2)] DeepVCP: An End-to-End Deep Neural Network for 3D Point Cloud Registration. [**`pc.`**]

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Golyanik_Accelerated_Gravitational_Point_Set_Alignment_With_Altered_Physical_Laws_ICCV_2019_paper.pdf)] Accelerated Gravitational Point Set Alignment with Altered Physical Laws. [**`pc.`**]

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)] Deep Closest Point: Learning Representations for Point Cloud Registration. [**`pc.`**]

[[ICCV](http://openaccess.thecvf.com/content_ICCV_2019/papers/Prokudin_Efficient_Learning_on_Point_Clouds_With_Basis_Point_Sets_ICCV_2019_paper.pdf)] Efficient Learning on Point Clouds with Basis Point Sets. [[code](https://github.com/sergeyprokudin/bps)] [ **`pc.`**]

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

[[ICRA](https://arpg.colorado.edu/papers/hmrf_icp.pdf)] Robust low-overlap 3-D point cloud registration for outlier rejection. [[matlab](https://github.com/JStech/ICP)] [**`pc.`**]

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

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf)] 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. [[tensorflow](https://github.com/yewzijian/3DFeatNet)] [**`pc.`**]

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper.pdf)] Efficient Global Point Cloud Registration by Matching Rotation Invariant Features Through Translation Search. [**`pc.`**]

[[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Benjamin_Eckart_Fast_and_Accurate_ECCV_2018_paper.pdf)] HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration. [**`pc.`**]

[[ECCV](https://link.springer.com/chapter/10.1007/978-3-030-01216-8_4)] Robust Image Stitching with Multiple Registrations.


[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2773482)] Guaranteed Outlier Removal for Point Cloud Registration with Correspondences. [**`pc.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2730205)] Collocation for Diffeomorphic Deformations in Medical Image Registration. [**`medi.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2748125)] Hierarchical Sparse Representation for Robust Image Registration.

[[TPAMI](https://doi.org/10.1109/TPAMI.2017.2654245)] Multiresolution Search of the Rigid Motion Space for Intensity-Based Registration. 

[[3DV](https://arxiv.org/abs/1808.00671)] PCN: Point Completion Network. [[tensorflow](https://github.com/TonythePlaneswalker/pcn)] [**`pc.`** ] 🔥

[[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460825)] Robust Generalized Point Cloud Registration Using Hybrid Mixture Model. [**`pc.`**]

[[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461049)] A General Framework for Flexible Multi-Cue Photometric Point Cloud Registration. [**`pc.`**]

[[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593839)] Dynamic Scaling Factors of Covariances for Accurate 3D Normal Distributions Transform Registration. [**`pc.`**]

[[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593558)] Robust Generalized Point Cloud Registration with Expectation Maximization Considering Anisotropic Positional Uncertainties. [**`pc.`**]

[[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594514)] PCAOT: A Manhattan Point Cloud Registration Method Towards Large Rotation and Small Overlap. [**`pc.`**]




## 2017

[[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zeng_3DMatch_Learning_Local_CVPR_2017_paper.pdf)] 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions. [[code](https://github.com/andyzeng/3dmatch-toolbox)] [**`pc.`** **`data.`** ] 🔥 ⭐

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

[[AAAI](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14751)] An Artificial Agent for Robust Image Registration.

[[AAAI](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14188)] Non-Rigid Point Set Registration with Robust Transformation Estimation under Manifold Regularization.

[[ICRA](https://ieeexplore.ieee.org/document/7989664)] Using 2 point+normal sets for fast registration of point clouds with small overlap. [**`pc.`**]

[[TPAMI](https://doi.org/10.1109/TPAMI.2016.2630687)] Image Registration and Change Detection under Rolling Shutter Motion Blur. 

[[TPAMI](https://doi.org/10.1109/TPAMI.2016.2567398)] Hyperbolic Harmonic Mapping for Surface Registration.

[[TPAMI](https://doi.org/10.1109/TPAMI.2016.2598344)] Randomly Perturbed B-Splines for Nonrigid Image Registration.





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

## 1. Papers

### 1.1. Overview & Survey Papers

#### Medical Image

1. A. Sotiras, et.al., [“Deformable medical image registration: A survey,”]( https://ieeexplore.ieee.org/document/6522524 ) 2013.

2. N. J. Tustison, et.al., [“Learning image-based spatial transformations via convolutional neural networks : A review,” ]( https://www.sciencedirect.com/science/article/abs/pii/S0730725X19300037 )2019.
3. G. Haskins,et.al. [“Deep Learning in Medical Image Registration: A Survey,” ]( https://arxiv.org/pdf/1903.02026.pdf )2019.

4. N. Tustison, et.al., [“Learning image-based spatial transformations via convolutional neural networks: A review,”]( https://www.sciencedirect.com/science/article/abs/pii/S0730725X19300037 )2019.




### 1.2. Key Algorithms



---

## 2. Datasets & Competitions

### 2.1. Datasets

#### Medical Image 

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

[**NWPU-RESISC45**](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html)

[**DOTA**](http://captain.whu.edu.cn/DOTAweb/index.html)

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

[[UWA Dataset](http://staffhome.ecm.uwa.edu.au/~00053650/databases.html)] . [**`pc.`**]

[[ASL Datasets Repository(ETH)](https://projects.asl.ethz.ch/datasets/doku.php?id=home)] This site is dedicated to provide datasets for the Robotics community  with the aim to facilitate result evaluations and comparisons. [ **`pc.`** ]

[[3D Match](http://3dmatch.cs.princeton.edu/)] Keypoint Matching Benchmark, Geometric Registration Benchmark, RGB-D Reconstruction Datasets. [**`pc.`** ]



### 2.2. Competitions


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


## 4. Books & Tutorials    

### 4.1. Books

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
### 4.2. Tutorials

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



### 4.3. Blogs

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



#### 点云配准

[点云配准算法的说明与流程介绍](https://blog.csdn.net/Ha_ku/article/details/797556232)

[几种点云配准算法的方法的介绍与比较](https://blog.csdn.net/weixin_43236944/article/details/881885323)

[三维点云用机器学习的方法进行处理](https://blog.csdn.net/u014636245/article/details/827559664)

[一个例子详细介绍了点云配准的过程](https://www.zhihu.com/question/34170804)

---

## 5. Courses/Seminars/Videos

### Courses

[**16-822: Geometry-based Methods in Vision**](http://www.cs.cmu.edu/~hebert/geom.html)

[VALSE 2018] [Talk: 2017以来的2D to 3D](https://zhuanlan.zhihu.com/p/38611920) by 吴毅红



### Workshops

[WBIR - International Workshop on Biomedical Image Registration](https://dblp.org/db/conf/wbir/index.html)

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



#### Point Cloud

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



#### Point Cloud

IEEE旗下的TPAMI，TIP等，还有SIAM Journal Image Sciences，Springer那边有IJCV













---

## **How to contact us**

We have QQ Group [【配准萌新交流群】](https://jq.qq.com/?_wv=1027&k=5r40AsF) （群号 869211738）

and Wechat Group 【配准交流群】（**已满员**） for comunications.



**More items will be added to the repository**.
Please feel free to suggest other key resources by opening an issue report,
submitting a pull request, or dropping me an email @ (im.young@foxmail.com).
Enjoy reading!





## Acknowledgments

Many thanks ❤️ to the other awesome list:

- **[Yochengliu](https://github.com/Yochengliu)**  [awesome-point-cloud-analysis](https://github.com/Yochengliu/awesome-point-cloud-analysis) ,
- **[NUAAXQ](https://github.com/NUAAXQ)**  [awesome-point-cloud-analysis-2022](https://github.com/NUAAXQ/awesome-point-cloud-analysis-2022)  
- **[hoya012](https://github.com/hoya012)**  [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)     
- [**JunMa11**](https://github.com/JunMa11) [MICCAI-OpenSourcePapers](https://github.com/JunMa11/MICCAI-OpenSourcePapers)
- **[Amusi](https://github.com/amusi)**  [awesome-object-detection](https://github.com/amusi/awesome-object-detection) 
- [**youngfish42**](https://github.com/youngfish42)  [Awesome-Federated-Learning-on-Graph-and-Tabular-Data](https://github.com/youngfish42/Awesome-Federated-Learning-on-Graph-and-Tabular-Data)  
- [**yzhao062**](https://github.com/yzhao062/)  [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources)
