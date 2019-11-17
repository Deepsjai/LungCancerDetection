# Introduction

Cancer is the second leading cause of death globally and is responsible for an estimated 9.6 million deaths in 2018. Lung cancer is the leading cause of cancer death in the United States with an estimated 160,000 deaths in the past year [[1]](#ref1). Early detection of cancer hence plays a key role in diagnosis which, in turn, improves the long-term survival rates. 

There are several barriers to the early detecting of cancer, such as a global shortage of radiologists. In addition to the shortage, detecting malignant tumors in X-rays can be difficult and challenging even for experienced radiologists. This time-consuming process typically leads to fatigue-based diagnostic errors and discrepancies [[2]](#re2).

Our project focuses on detecting the presence of malignant tumors in chest X-rays. In order to aid radiologists around the world, we propose to exploit supervised and unsupervised Machine Learning algorithms for lung cancer detection. We aim to showcase ‘explainable’ models [[3]](#ref3) that could perform close to human accuracy levels for cancer-detection. We envision our models being used to assist radiologists and scaling cancer detection to overcome the lack of diagnostic bandwidth in this domain. We can also potentially export our models to personal devices, which would allow for easier, cheaper and more accessible cancer detection.  

____

# Supervised Learning

## Approach: 
We use a transfer learning approach to perform supervised binary classification of images as 'benign' or 'malignant' based on the presence of malignant tumors. 

## Reason behind adopting this approach:

It is known that tumors are of different shapes and sizes and can occur at different locations, which makes their detection challenging [[5]](#ref5). In addition to this, deep learning approaches have been showing expert-level performance in medical image interpretation tasks in the recent past (for eg., Diabetic Retinopathy [[6]](#ref6)). This can be attributed to both - availability of large labeled data sets and the ability of deep neural networks to extract complex features from within the image. 

It would be tedious (and maybe near impossible) to hand-design the features that one would need to build models for this task. This, in combination with the fact that we were dealing with a dataset containing a significantly smaller amount of images directly points to using a transfer learning approach where we initialize the parameters from a network pre-trained on ImageNet data and modify the final fully connected layer of the pre-trained network to a new fully-connected layer producing 2 responses indicative of the predicted probabilities of the two classes.

## What’s new in our approach:

The overall architecture of feature_extraction + grad_cam visualization + Augmentation via VAEs is new and has not been approached on a medical image dataset to the best of our knowledge. \
If our approach can show improved results, it could mean that we do not necessarily have to collect a large amount of data at all times and would be able to manage with smaller datasets. 

### Proposed System Architecture

![](./images/image7.jpg)

Fig 1. System Architecture: The classifier is trained on the training dataset and the generated data from the Variational AutoEncoders. The model classifies a test X-ray as benign or malignant and highlights the region that contributes most to the classification. 

_____

# Grad CAM

Among the most important areas of research in deep learning today is that of interpretability, i.e, being able to demystify the black-box nature (owing to its non-convex nature) of a neural network and identify the key reasons for making its predictions. Various approaches have been proposed to help with this exercise, the most recent of which involves gradient-based class activation mappings that highlight the specific pixels (or regions) of an image that most strongly activate a certain class of the model’s prediction. Abbreviated as Grad-CAM, this approach has become a universally accepted yardstick for interpretability in the deep learning research community across a wide range of tasks such as image classification, object detection, image captioning and visual question answering. 

This becomes a particularly relevant addition to a medical diagnostic tool considering the serious implications of algorithmic decision making in this domain. It is essential to build trust in the algorithms among doctors and patients alike. Critically, it also sheds light on the imperfections of a trained model when it makes incorrect predictions or when it makes the right predictions for the wrong reasons. While somewhat intellectually dissatisfying, it shouldn’t surprise us that these cases are plenty in number because the training paradigm in deep learning problems simply maps input data to output labels, with no scope for detailed reasoning on the causal relationships behind this mapping. These heatmaps generated from Grad-CAM are an important step in the direction of making our model’s predictions more trustworthy, and in this domain, would aid a radiologist in examining potentially relevant areas in the X-ray images that are likely to be of diagnostic importance.

We demonstrate a few applications of Grad-CAM to our problem and showcase its usefulness (and occasional unreliability) in the following examples.

![](./images/image3.png) ![](./images/image4.png)

Fig 2. On the left, the original X-ray image that’s been (correctly) classified as malignant. On the right, the Grad-CAM heatmap that points to the precise region in the X-ray where there’s a clumping of cells that explains the prediction of malignancy. 

________

# Unsupervised Learning
In the clinical setting, it becomes extremely important to train a model that can handle a range of variations in the patient’s X-ray scan. However in the modern-day world with genetic variations and evolution taking place at an ever-growing rate, it becomes nearly impossible to obtain all possible variations of input. In addition to this one of the biggest challenges in the medical field is the lack of sufficient image data, which are laborious and costly to obtain. Data augmentation is one such technique that is leveraged to increase the variability of the dataset, thus reducing the risk of overfitting. Conventional transformation methods (eg: flip, rotation) can be used to augment our training corpus, but their outputs are highly dependent on the original data. Hence we propose to make use of an unsupervised technique of generating new samples having similar properties as that of the dataset. 

The Variational Autoencoder (VAE) is one such generative model that estimates the probability density function of the training dataset. VAE is an architecture comprising of an encoder and a decoder, which is trained to minimise the reconstruction error between the encoded-decoded data and the initial data. The encoder projects each input datapoint onto a latent space that follows a normal distribution. Thus it converts the input into a d-dimensional latent vector that can be sampled with mean  and standard deviation  through reparametrization. The decoder then decodes these latent representations and reconstructs the input data. The loss function of the variational autoencoder is the sum of the reconstruction loss and the regularizer.


Eq 1. Loss function of a Variational Autoencoder


The first term is the reconstruction loss, or the expected negative log-likelihood of the i-th datapoint. The expectation is taken with respect to the encoder’s distribution over the representations. This term encourages the decoder to learn to reconstruct the data. The second term is a regularizer which in our case is the Kullback-Leibler divergence between the encoder’s distribution and the standard Gaussian distribution.

We carried out our experiments on two VAE architectures : a fully connected linear VAE and a deep neural network VAE having the following architectures.

![](./images/image5.png)
Fig 3. Architecture of the Variational AutoEncoders used.

__________

# Dataset

We used the small CheXpert Chest radiograph dataset [[7]](#ref7) to build our initial dataset of images. To build our dataset, we sampled data corresponding to the presence of a ‘lung lesion’ which was a label derived from either the presence of “nodule” or “mass” (the two specific indicators of lung cancer). 

The initial (unaugmented) dataset:

#### Train:
Benign images (Negative class): 6488 images \
Malignant (Positive class): 6287 images

#### Validation:
Benign images (Negative class): 1500 images \
Malignant (Positive class): 1450 images

#### Test:
Benign (Negative class): 1500 images \
Malignant (Positive class): 1449 images


In the training phase, we treated all images with a transformations to augment our data by performing random resized crop and lateral inversions with a 50% probability. 

In addition to the above all images were normalized using the channel-wise mean and standard deviation values computed on the ImageNet dataset. 

___________

# Results

_______

# References

<a name="ref1"></a> 1. Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T. et.al., M. P. (2017). Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. \
<a name="ref2"></a> 2. Yongsik Sim,  Myung Jin Chung et al. Deep Convolutional Neural Network–based Software Improves Radiologist Detection of Malignant Lung Nodules on Chest Radiographs, Radiology, 2019 \
<a name="ref3"></a> 3. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, 2017, pp. 618-626. \
<a name="ref4"></a> 4. Kingma P, Welling M., An Introduction to Variational Autoencoders, arXiv:1906.02691. \
<a name="ref5"></a> 5. Ardila, D., Kiraly, A.P., Bharadwaj, S. et al. End-to-end lung cancer screening with three-dimensional deep learning on low-dose chest computed tomography. Nat Med 25, 954–961 (2019). \
<a name="ref6"></a> 6. Gulshan V, Peng L, Coram M, et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA. 2016.\
<a name="ref7"></a> 7. Irvin, Jeremy & Rajpurkar, Pranav & Ko, Michael & Yu, Yifan & Ciurea-Ilcus, Silviana & Chute, Chris & Marklund, Henrik & Haghgoo, Behzad & Ball, Robyn & Shpanskaya, Katie & Seekins, Jayne & Mong, David & Halabi, Safwan & Sandberg, Jesse & Jones, Ricky & Larson, David & Langlotz, Curtis & Patel, Bhavik & Lungren, Matthew & Ng, Andrew. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. 





