Deep Learning Applicative Solutions

This repository contains an extensive set of deep learning solutions addressing complex and practical questions across various datasets and architectures. Each solution demonstrates advanced techniques and best practices in neural network implementation, optimization, and evaluation.

Topics Covered

Neural Networks

Optimizers: SGD, MBGD, AdaGrad, Adam

Regularization techniques

Initialization methods

Deep Learning Models

Autoencoders (basic and denoising)

Convolutional Neural Networks (CNNs)

Recurrent Neural Networks (RNNs) and LSTMs

Network Architecture Search (NAS)

Deep Generative Models

Deep Belief Networks (DBNs)

Variational Autoencoders (VAEs)

Generative Adversarial Networks (GANs), including DCGANs and CycleGANs

Representation Learning

Unsupervised pretraining

Transfer learning and domain adaptation

Distributed representation

Discovering underlying causes

Laboratory Practices

Implementations of Autoencoder, CNN, LSTM, DBN, GANs (variants)

Transfer Learning, Graph Neural Networks (GNNs), and adversarial losses

Questions Addressed

Dry Bean Dataset Classification

Implemented a Multi-Layer Perceptron (MLP) for classification.

Autoencoder Design

Built a fully connected autoencoder and denoising autoencoder using the SVHN dataset.

Greedy Layer-Wise Pretraining

Conducted experiments on CIFAR-10 with convolutional blocks and autoencoders.

LSTM for Time Series Prediction

Forecasted airplane passenger data using LSTM, evaluating MSE, RMSE, and MAPE.

Variational Autoencoder with UNet Connections

Trained and compared VAEs and autoencoders on the FaceMask dataset.

DCGAN Implementation

Designed and evaluated DCGANs on the CelebA dataset, incorporating least-squares GAN loss.

CycleGAN Implementation

Enhanced CycleGAN with spatial transformations, semi-supervised learning, and adversarial robustness.

StyleGAN

Generated realistic images and performed feature disentanglement and interpolation.

Transfer Learning

Fine-tuned InceptionNet for satellite image classification and saliency map generation.

Graph Neural Network (GNN)

Classified MNIST images using a GNN with GATConv layers and compared with CNN performance.

SegNet for Image Segmentation

Trained SegNet on the Pascal VOC dataset and compared performance with lightweight backbones.

Highlights

Comprehensive Codebase: Includes implementation of cutting-edge techniques such as adversarial losses, annealing schedules, and spatial transformations.

Evaluation Metrics: Thorough assessments using SSIM, PSNR, FID, Mean IoU, confusion matrices, and more.

Visualization: T-SNE plots, saliency maps, and training loss curves for in-depth analysis.

Optimization: Techniques like hyperparameter tuning, greedy pretraining, and adversarial training ensure robust performance.

Repository Structure

Q1_MLP_DryBean/ – Classification using MLP

Q2_Autoencoder_SVHN/ – Fully connected autoencoder and denoising autoencoder

Q3_GreedyPretraining_CIFAR10/ – Layer-wise pretraining experiments

Q4_LSTM_AirplanePassengers/ – LSTM for time series forecasting

Q5_VAE_UNet_FaceMask/ – Variational Autoencoder with UNet connections

Q6_DCGAN_CelebA/ – DCGAN implementation and comparison

Q7_CycleGAN/ – Enhanced CycleGAN with spatial transformations and semi-supervised learning

Q8_StyleGAN/ – StyleGAN experiments and feature disentanglement

Q9_TransferLearning_SatelliteImages/ – Transfer learning with InceptionNet

Q10_GNN_MNIST/ – GNN for MNIST classification

Q11_SegNet_PascalVOC/ – SegNet implementation and evaluation

Usage

Clone the repository:

git clone https://github.com/yourusername/deep-learning-applications.git

Navigate to the desired question folder and run the provided scripts:

cd Q1_MLP_DryBean
python train.py

Contributions

Contributions are welcome! Please raise an issue or create a pull request for enhancements, bug fixes, or new implementations.
