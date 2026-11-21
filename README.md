# GenAI CV Homework
- Author: Vladyslav Cherniuk

First of all, I'd like to mention that all the models I tried are intended to fit into my laptop GPU - RTX 5070 Mobile. So the hyperparameters and model architectures were picked having the 8 GB VRAM in mind.

### 1. RealNVP

I have used the RealNVP example from the lecture and adapted it for working with MNIST dataset images:
- Removed unnecessary code;
- Loaded the MNIST dataset;
- Changed the layer sizes: 784 (28*28) input and output for flattened images, and 1024 hidden, so that it is greater than the number of pixels in the image;
- Changed the mask: I made a mask that covers to transform only the top half of the image (14*28), but the mask is flattened, so it's just 784/2 first pixels;
- Changed visualizations: as generated moon dataset has only 2 dimensions, it was easier to visualize. To visualize the 784-dimensional data I used PCA, fitted on each set of observations separately, when comparing generated and original distributions. It was made this way because fitting PCA to only one distribution and transforming both resulted in one of the distributions being just a point. For input-output layers it's different because the latent space is 0-mean with std 1, so PCA for this kind of data would just show a small point cloud that is impossible to compare to the other data;
- Added CUDA tensors utilization: the example was utilizing only CPU, so the training was quite slow, I've updated the code to move the data to the GPU;
- Added visualizations of specific generated data samples;

These actions led to the following results:

![RealNVP_Baseline.png](images/RealNVP_Baseline.png)

![RealNVP_Baseline_Loss.png](images/RealNVP_Baseline_Loss.png)
Loss: -4.8782

Then I tried experimenting with the hyperparameters:
- Increasing/decreasing the hidden layer size;
- Increasing/decreasing the number of coupling layers;
- Adapting the learning rate for larger models;

I didn't succeed in making high hidden layer size and deep overall network with big amount of coupling layers simultaneously, so I tried adapting them one or another. Having more than 12 coupling layers or more than 2048 nodes in the hidden layer resulted in either quick overfitting or great numerical metrics and poor overlap with the original data. For example, this is the model  with 2084 nodes in the hidden layer and 16 coupling layers.

![RealNVP_16_Coupling_Layers.png](images/RealNVP_16_Coupling_Layers.png)

![RealNVP_16_Coupling_Layers_Loss.png](images/RealNVP_16_Coupling_Layers_Loss.png)
Loss: -8.6234

Then I tried changing the architecture: adding one hidden layer inside the coupling layers. So the best model with was the one with two hidden layers in the coupling layer with 2048 nodes each and 12 coupling layers.

![RealNVP_2_Hidden_Layers.png](images/RealNVP_2_Hidden_Layers.png)

![RealNVP_2_Hidden_Layers_Loss.png](images/RealNVP_2_Hidden_Layers_Loss.png)
Loss: -7.4559

Then I just trained the same model on MNIST Fashion dataset and got the following results.

![RealNVP_Fashion.png](images/RealNVP_Fashion.png)

![RealNVP_Fashion_Loss.png](images/RealNVP_Fashion_Loss.png)

Loss: -6.6809


### 2. VAE

The same way as for RealNVP, I have used the VAE example from the lecture and adapted it for working with MNIST dataset images:
- Removed unnecessary code;
- Loaded the MNIST dataset;
- Changed the layer sizes: 784 (28*28) input and output for flattened images, and 1024 hidden, so that it is greater than the number of pixels in the image;
- Changed visualizations to use PCA;
- Added visualizations of specific generated data samples;

The resulting model was very poor and didn't have a good overlay:

![VAE_Baseline.png](images/VAE_Baseline.png)

I have increased the size of the hidden layers to 4096 and got the fast KL loss convergence. The loss was just 0 without updates.

![VAE_Increased_Hidden_Layers.png](images/VAE_Increased_Hidden_Layers.png)

- Total Loss: 0.0674
- Reconstruction Loss: 0.0673
- KL Divergence Loss: 0.0003

Then I tried to experiment with the hidden layer and latent dimension
sizes, the best model had hidden layer size 8192 and latent dimension size 256. 

![VAE_Latend_Dimension_256.png](images/VAE_Latent_Dimension_256.png)

- Total Loss: 0.0494
- Reconstruction Loss: 0.0208
- KL Divergence Loss: 0.0573

Then I've added a RealNVP part that was trained simultaneously with the encoder and the decoder. I've experimented with different structures, layer sizes, coupling layer counts, and the best results were for 8 coupling layers of size 512 with 2 hidden layers. I also changed the VAE hidden layer size to 4096 as the model did not fit into my VRAM. Also, I've increased the initial number of epochs 200 to be 400 as the model had become bigger.

![VAE_Coupling_Layers.png](images/VAE_Coupling_Layers.png)

- Total Loss: 0.0451
- Reconstruction Loss: 0.0255
- KL Divergence Loss: 0.0391

I also tried adjusting $\beta$ parameter for $\beta$-VAE. In the example, it was 0.5 initially. I tried 0.2 - 3, but all of them didn't improve the model's performance, so I just kept 0.5, keeping the previous model as the best one.

For 0.2 the latent space just did not converge keeping high KL loss and leading to poor generation.

![VAE_Beta_0_2.png](images/VAE_Beta_0_2.png)

- Total Loss: 0.0372
- Reconstruction Loss: 0.0161
- KL Divergence Loss: 0.1054

For 0.8 and 1.5 the KL divergence did improve but the total loss was still quite high. For example, $\beta$ = 0.8.

![VAE_Beta_0_8.png](images/VAE_Beta_0_8.png)

- Total Loss: 0.0585
- Reconstruction Loss: 0.0297
- KL Divergence Loss: 0.0359

$\beta$ = 3 just led to the KL divergence loss 0, so the encoder and decoder just didn't train, resulting in poor generation.

![VAE_Beta_3.png](images/VAE_Beta_3.png)

- Total Loss: 0.0733
- Reconstruction Loss: 0.0613
- KL Divergence Loss: 0.0040

Then I tried to train the same best model for MNIST Digits on MNIST Fashion dataset. It did not perform well on that dataset initially, not learning the distribution at all.

![VAE_Fashion_Baseline.png](images/VAE_Fashion_Baseline.png)

- Total Loss: 0.0495
- Reconstruction Loss: 0.0429
- KL Divergence Loss: 0.0133

So I experimented a little with specific architecture for that dataset and found out that nothing could improve the model as much as the batch size. It turns out that this big batch size averaged gradients too much and the model just didn't learn the specific features. Increasing the learning rate could help, but it was not the case here. So I decreased the batch size from 8192 to 512.

![VAE_Fashion_Batch_512.png](images/VAE_Fashion_Batch_512.png)

- Total Loss: 0.0351
- Reconstruction Loss: 0.0184
- KL Divergence Loss: 0.0333

I also tried setting different $\beta$ and value 0.2 has improved the total loss and generation a little more.

![VAE_Fashion_Beta_0_2.png](images/VAE_Fashion_Beta_0_2.png)

- Total Loss: 0.0264
- Reconstruction Loss: 0.0159
- KL Divergence Loss: 0.0525
