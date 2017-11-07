# Neural Networks for Low Level Image Processing

Until recently, machine learning (ML) or neural networks (NN) are mainly used in high level vision tasks, such as image segmentation, object recognition and detection. Low level image processing such as denoising, demosaicing, white balance still mainly rely on signal processing based methods which uses expert designed filters. There are usually a long list of filters in the whole processing pipeline which is run on a dedicated ISP chip. In the past one or two years, there are two new trends. One trend is that more and more researchers propose to apply NN for low level image processing and achieved fascinating performance in term of image quality and processing speed. The other trend is that neural network chip becomes more and more popular at various mobile platform, such as the latest Apple A11 Bionic chip and Huawei Kirin 970 chip. I believe in the near further, NN based methods will play important roles at some low level image processing tasks also some ISP chip may include some NN computing units.

This is a personal collection of works using neural networks for low level image processing. The list will be regularly updated. You are welcome to contribute. Papers of significance are marked in **bold**. My comments are marked in *italic*.

## Table of Contents

  * [Review and comments](#review-and-comments)
  * [Color constancy](#color-constancy)
  * [Denoising](#denoising)
  * [Demosaicing](#demosaicing)
  * [Superresolution](#superresolution)
  * [Automatic adjustment](#automatic-adjustment)
  * [Artefacts removal](#artefacts-removal)
  * [Pipeline](#pipeline)
  * [Image quality evaluation](#image-quality-evaluation)
  * [Others](#others)

## Review and comments

  * [Deep, Deep Trouble](https://sinews.siam.org/Details-Page/deep-deep-trouble)
    * *This is an interesting comments from Michael Elad about the impact of Deep Learning on image processing, mathematics and humanity.*
  * [Deep learning for image/video processing](https://www.slideshare.net/yuhuang/deep-learning-for-image-video-processing)
    * *Dr. Huang Yu's summary of DL on image/video processing.*

## Color constancy

  * [Deep Specialized Network for Illuminant Estimation](http://mmlab.ie.cuhk.edu.hk/projects/illuminant_estimation.html) (ECCV, 2016, CUHK)
  * [Single and Multiple Illuminant Estimation Using Convolutional Neural Networks](https://arxiv.org/abs/1508.00998) (TIP, 2017, Italy) 
    * *A three-stage method for illuminant estimation from RAW images*
  * [Recurrent Color Constancy](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qian_Recurrent_Color_Constancy_ICCV_2017_paper.pdf) (ICCV, 2017, TUT)
    * *Recurrent network for temporal color constancy using multiple frames instead of single frame*
  * [FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling](http://openaccess.thecvf.com/content_cvpr_2017/papers/Hu_FC4_Fully_Convolutional_CVPR_2017_paper.pdf) (CVPR, 2017, THU)
    * *CNN based method using image patches*
   * [Natural Image Illuminant Estimation via Deep Non-negative Matrix Factorisation](http://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2016.1058) (IET-IP, 2017, OceanUofChina)


## Denoising

  * [**Image denoising with multi-layer perceptrons**](https://arxiv.org/abs/1211.1544) (arXiv, 2012)
    * *This is the very first paper using NN to image denoising tasks.*
  * [Can a Single Image Denoising Neural Network Handle All Levels of Gaussian Noise?](https://www.semanticscholar.org/paper/Can-a-Single-Image-Denoising-Neural-Network-Handle-Wang-Morel/c0387d184c2201eb1811094ba259380b5a83b6a4) (SPL, 2014, France)
    * *This paper proposal a way to apply NN on image denoising with different noise level of Gaussian noise.*
  * [Deep convolutional architecture for natural image denoising](http://ieeexplore.ieee.org/document/7341021/) (WCSP, 2015, ZJU)
  * [Learning Deep CNN Denoiser Prior for Image Restoration](https://arxiv.org/abs/1704.03264) (arXiv, 2017, HKPolyU)
  * [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](http://ieeexplore.ieee.org/document/7839189/) (TIP, 2017, HKPolyU)
  * [FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising](https://arxiv.org/abs/1710.04026) (arXiv, 2017, HKPolyU)
  * [Image Restoration: From Sparse and Low-Rank Priors to Deep Priors](http://ieeexplore.ieee.org/document/8026108/) (IPM, 2017, HKPolyU) 
    * *This is a review paper.*
  * [Dilated Deep Residual Network for Image Denoising](https://arxiv.org/abs/1708.05473) (arXiv, 2017, SIU)
  * [IDEAL: Image DEnoising AcceLerator](http://www.eecg.toronto.edu/~mostafam/files/IDEAL_Image_DEnoising_AcceLerator.pdf) (ACM, 2017, UToronto&Algolux) 
    * *A NN based approximations of BM3D on an NN accelerator*

## Demosaicing

  * [Demosaicing using artificial neural networks](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/3962/1/Demosaicking-using-artificial-neural-networks/10.1117/12.382904.short?SSO=1) (SPIE, 2000)
  * [A multilayer neural network for image demosaicking](https://www.semanticscholar.org/paper/A-multilayer-neural-network-for-image-demosaicking-Wang/4264e3560bf59bfdb9f262a9787187257a1ce75f) (ICIP, 2012)
  * [Deep Joint Demosaicking and Denoising](https://groups.csail.mit.edu/graphics/demosaicnet/) (Siggraph, 2016, MIT&Adobe) 
    * *This paper propose to use more difficult patches for the training.*
    * *Data and code are available on Github.*

## Automatic adjustment

From the publications, we can find that Adobe has done a lot of work pushing the usage of machine learning in low-level image processing especially automatic photo adjustment. 

  * [Learning Photographic Global Tonal Adjustment with a Database of Input / Output Image Pairs](http://people.csail.mit.edu/vladb/photoadjust/) (CVPR, 2011, MIT&Adobe) 
    * *Adjustment personalization*
  * [Automatic Photo Adjustment Using Deep Neural Networks](https://sites.google.com/site/homepagezhichengyan/home/dl_img_adjust) (ACM, 2016, UIUC&Adobe&Microsoft) 
    * *This technique is more accurate and supports local edits.*

## Superresolution

Super-resolution is one of the areas that NN has been applied extensively and achieved great success.

  * [Deep Networks for Image Super-Resolution with Sparse Prior](http://ieeexplore.ieee.org/document/7410407/) (ICCV, 2015, UIUC)
  * [Image superresolution using deep convolutional networks](https://ieeexplore.ieee.org/document/7115171/) (TPAMI, 2015, HKUST)
  * [Accurate image super-resolution using very deep convolutional networks](https://arxiv.org/abs/1511.04587) (CVPR, 2016, Seoul Univ.)
  * [**Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf) (CVPR, 2016, ICL) 
    * *An order of magnitude faster than previous CNN-based methods*
  * [Image Super-Resolution via Deep Recursive Residual Network](http://cvlab.cse.msu.edu/project-super-resolution.html) (CVPR, 2017, MSU)
  * [Photo-realistic single image super-resolution using a generative adversarial network](https://arxiv.org/abs/1609.04802) (CVPR, 2017, Twitter)

## Artefacts removal

  * [Deep Generative Adversarial Compression Artifact Removal](https://arxiv.org/abs/1704.02518) (arXiv, 2017, UoFlorence)
  * [Real-time Deep Video Deinterlacing](https://arxiv.org/abs/1708.00187) (arXiv, 2017, CUHK) 

## Pipeline

  * [Learning the image processing pipeline](https://arxiv.org/abs/1605.09336) (arXiv, 2016, Stanford)
	  * *Propose to learn the filter parameters by ML*
  * [Learning Adaptive Parameter Tuning for Image Processing](https://arxiv.org/abs/1610.09414) (arXiv, 2016, UCLA&Nvidia)	  

## Image quality evaluation

  * [Deep Learning for Blind Image Quality Assessment](http://www.ivl.disco.unimib.it/activities/deep-image-quality/) (2017, Unimib)

## Others

  * [Fast Image Processing with Fully-Convolutional Networks](https://www.youtube.com/watch?v=eQyfHgLx8Dc&feature=youtu.be) (ICCV, 2017, Intel) 
    * *Operator approximation to accelerate image processing tasks*
  * [Learning Proximal Operators:Using Denoising Networks for Regularizing Inverse Imaging Problems](https://arxiv.org/pdf/1704.03488.pdf) (arXiv, 2017, TUM) 
    * *This paper proposes a way to use NN for the the general fidelity plus regularization optimization problem which may find applications in many problems.*
