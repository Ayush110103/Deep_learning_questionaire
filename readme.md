# <a name="_sansspz3ln6c"></a>**Deep Learning Applicative Solutions**
This repository contains an extensive set of deep learning solutions addressing complex and practical questions across various datasets and architectures. Each solution demonstrates advanced techniques and best practices in neural network implementation, optimization, and evaluation.
## <a name="_cgs3t14h35jg"></a>**Topics Covered**
1. **Neural Networks**
   1. Optimizers: SGD, MBGD, AdaGrad, Adam
   1. Regularization techniques
   1. Initialization methods
1. **Deep Learning Models**
   1. Autoencoders (basic and denoising)
   1. Convolutional Neural Networks (CNNs)
   1. Recurrent Neural Networks (RNNs) and LSTMs
   1. Network Architecture Search (NAS)
1. **Deep Generative Models**
   1. Deep Belief Networks (DBNs)
   1. Variational Autoencoders (VAEs)
   1. Generative Adversarial Networks (GANs), including DCGANs and CycleGANs
1. **Representation Learning**
   1. Unsupervised pretraining
   1. Transfer learning and domain adaptation
   1. Distributed representation
   1. Discovering underlying causes
1. **Laboratory Practices**
   1. Implementations of Autoencoder, CNN, LSTM, DBN, GANs (variants)
   1. Transfer Learning, Graph Neural Networks (GNNs), and adversarial losses
## <a name="_npqjssmpuja7"></a>**Questions Addressed**
1. **Dry Bean Dataset Classification**
   1. Implemented a Multi-Layer Perceptron (MLP) for classification.
1. **Autoencoder Design**
   1. Built a fully connected autoencoder and denoising autoencoder using the SVHN dataset.
1. **Greedy Layer-Wise Pretraining**
   1. Conducted experiments on CIFAR-10 with convolutional blocks and autoencoders.
1. **LSTM for Time Series Prediction**
   1. Forecasted airplane passenger data using LSTM, evaluating MSE, RMSE, and MAPE.
1. **Variational Autoencoder with UNet Connections**
   1. Trained and compared VAEs and autoencoders on the FaceMask dataset.
1. **DCGAN Implementation**
   1. Designed and evaluated DCGANs on the CelebA dataset, incorporating least-squares GAN loss.
1. **CycleGAN Implementation**
   1. Enhanced CycleGAN with spatial transformations, semi-supervised learning, and adversarial robustness.
1. **StyleGAN**
   1. Generated realistic images and performed feature disentanglement and interpolation.
1. **Transfer Learning**
   1. Fine-tuned InceptionNet for satellite image classification and saliency map generation.
1. **Graph Neural Network (GNN)**
   1. Classified MNIST images using a GNN with GATConv layers and compared with CNN performance.
1. **SegNet for Image Segmentation**
   1. Trained SegNet on the Pascal VOC dataset and compared performance with lightweight backbones.
## <a name="_f5cse4ul6mdl"></a>**Highlights**
- **Comprehensive Codebase**: Includes implementation of cutting-edge techniques such as adversarial losses, annealing schedules, and spatial transformations.
- **Evaluation Metrics**: Thorough assessments using SSIM, PSNR, FID, Mean IoU, confusion matrices, and more.
- **Visualization**: T-SNE plots, saliency maps, and training loss curves for in-depth analysis.
- **Optimization**: Techniques like hyperparameter tuning, greedy pretraining, and adversarial training ensure robust performance.
## <a name="_h551xa6g35an"></a>**Repository Structure**
- Q1\_MLP\_DryBean/ – Classification using MLP
- Q2\_Autoencoder\_SVHN/ – Fully connected autoencoder and denoising autoencoder
- Q3\_GreedyPretraining\_CIFAR10/ – Layer-wise pretraining experiments
- Q4\_LSTM\_AirplanePassengers/ – LSTM for time series forecasting
- Q5\_VAE\_UNet\_FaceMask/ – Variational Autoencoder with UNet connections
- Q6\_DCGAN\_CelebA/ – DCGAN implementation and comparison
- Q7\_CycleGAN/ – Enhanced CycleGAN with spatial transformations and semi-supervised learning
- Q8\_StyleGAN/ – StyleGAN experiments and feature disentanglement
- Q9\_TransferLearning\_SatelliteImages/ – Transfer learning with InceptionNet
- Q10\_GNN\_MNIST/ – GNN for MNIST classification
- Q11\_SegNet\_PascalVOC/ – SegNet implementation and evaluation

