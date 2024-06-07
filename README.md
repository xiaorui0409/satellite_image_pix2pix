# Enhancing Image Quality with Pix2Pix

## Background
**Pix2Pix** is a conditional generative adversarial network renowned for its ability to perform image-to-image translation, preserving structural integrity while transferring styles. This makes it ideal for applications like converting sketches into photographs or black-and-white images into color.

## Potential Applications
### 1. **Enhancing Satellite Image Quality**
Using Pix2Pix, we aim to enhance the resolution of satellite imagery, often termed "super-resolution." This technique is crucial where high-resolution data is essential but restricted by sensor limitations or atmospheric conditions.

### 2. **Generating Forest Recovery Scenarios**
Our model can simulate forest recovery from pre- to post-recovery states, using training data to forecast recovery under various conditions, aiding in conservation efforts.

## Data Source
Due to restrictions on accessing the original dataset, we utilize the [VOCSegmentation dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.VOCSegmentation.html) to train our model. This dataset includes essential pairs of original images and segmentation masks. Our custom `VOCPairedDataset` class ensures each segmentation mask aligns with its corresponding real image for effective training.Below are sample images to demonstrate the required pairwise data structure for training set.
![Sample Training Data](Figure_1)

## Architecture
The generator of Pix2Pix is based on the **U-Net architecture**, featuring contracting and expanding blocks:
- **Contracting blocks** compress the input into a high-dimensional space.
- **Expanding blocks** reconstruct the image from this compressed representation.
- **Skip connections** preserve features lost during downsampling, enhancing feature transmission between blocks.
The image below, taken from the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al., 2015(https://arxiv.org/abs/1505.04597) depicts the U-Net architecture, demonstrating its contracting and then expanding processes.    
![Pix2Pix Training Data Architecture](path_to_image)


The **discriminator**  is based on the contracting path of the U-Net, assesses image realism using a "Discriminator Patch Gan," offering more granular feedback than typical cGANs.
For detailed insights, refer to the [Image-to-Image Translation with Conditional Adversarial Networks paper by Isola et al., 2017](https://arxiv.org/abs/1611.07004).

## Result Illustration
![Pix2Pix Training Data Architecture](path_to_image)

**Input Image:** A simplified abstract representation guiding the generative process.
**Ground Truth:** A high-resolution photograph showcasing the target output.
**Predicted Image:** Demonstrates the model's capability to transform basic inputs into detailed and realistic outputs.

### Enhancing Image Quality
In our project, blurry satellite images from sensor limitations or extreme weather provide minimal information, making it difficult to accurately assess environmental changes. Using a model trained on paired datasets, we enhance these blurry images to high-resolution, clear visuals, significantly improving our evaluation of forest recovery

### Forest Recovery Scenarios
By training on images showing forests before and after recovery, the model learns to transform these images, simulating potential recovery outcomes. This application aids in planning and resource allocation for forest conservation.

## Potential Challenges
- **Data Dependency:** Pix2Pix's effectiveness depends on high-quality, paired training datasets, which are scarce and often need to be synthesized.
- **Environmental Variability:** Changes in lighting, seasons, and weather conditions can hinder the model's ability to generalize across different scenarios without diverse training data.

