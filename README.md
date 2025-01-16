# Sea Turtle Segmentation

This repository contains the essential files for our project focused on segmenting sea turtles from images, specifically targeting the head, flippers, and carapace. This README provides guidance on the project’s deliverables and instructions on using the included files.

## Background

Sea turtles are an important part of marine ecosystems, and their conservation is crucial. Accurate segmentation of sea turtles in images can aid in monitoring their populations and health. This project aims to develop and optimize models for segmenting sea turtles from images.

## Dataset

The `SeaTurtleID2022` dataset, consisting of 8,729 images of 438 unique sea turtles, was used. The dataset spans 13 years and features high-quality annotations for body parts such as the head, flippers, and carapace. Its open-set splitting method ensures realistic evaluation of model generalization, mimicking real-world conditions. The dataset was preprocessed to address variations in lighting, scale, and noise. Key preprocessing steps included resizing images to standard input sizes (e.g., 128x128 for U-Net), augmentation (e.g., horizontal flipping), and normalization.

## Challenges

- **Data Quality**: Variability in underwater image quality and lighting conditions.
- **Occlusions**: Parts of the turtle may be occluded by other objects or by itself.
- **Class Imbalance**: Some parts of the turtle (e.g., flippers) may be underrepresented in the dataset.
- **Complex Backgrounds**: The presence of complex and varying backgrounds in the images.
- **Computational Resources**: High computational power required for training complex models.

## Solution

1. **Enhanced Preprocessing and Augmentation**

	Subtle data augmentations, including horizontal flips and rotation, were used to increase dataset diversity while maintaining realistic conditions. Normalization and resizing standardized input images, improving model training stability.

2. **Custom Loss Functions**

	We introduced a combined loss function, incorporating Cross-Entropy and Focal Tversky Loss, to address class imbalance. This improved segmentation accuracy, particularly for small and challenging regions like flippers and head boundaries.

3. **Integration of Attention Mechanisms**

	Attention mechanisms, such as CBAM and SEBlock, were integrated to guide the models in focusing on relevant spatial regions and important feature channels. This reduced the impact of background noise and enhanced detail segmentation.

4. **Model Optimizations**

	U-Net Enhancements: Adding ResNet-34 as a backbone improved feature extraction for challenging images, and CBAM refined spatial focus.
	
	Mask R-CNN Enhancements: Using FPN for multi-scale feature extraction improved instance segmentation in complex scenes. SEBlock was explored to enhance feature representation but required careful tuning to avoid overfitting.

5. **Computational Efficiency**

	Reduce training sample size when training Mask R-CNN model.
 	
	Efficient learning rate schedulers and adaptive optimizers like AdamW helped balance computational cost with performance. Early stopping ensured efficient training, reducing overfitting risks.


## Method

We implemented and compared two primary models: **U-Net** and **Mask R-CNN**, with additional enhancements:
	1.	U-Net Variants
	•	Baseline U-Net: Employed the standard architecture with cross-entropy loss and Adam optimizer.
	•	Enhanced U-Net: Introduced a combined loss function (Cross-Entropy + Focal Tversky Loss) to address class imbalance, and used ResNet-34 as a backbone for improved feature extraction.
	•	Attention Mechanisms: Incorporated CBAM (Channel and Spatial Attention Modules) to emphasize relevant spatial regions, significantly improving segmentation precision.
	2.	Mask R-CNN Variants
	•	Baseline Mask R-CNN: Utilized a ResNet-50 backbone paired with a Feature Pyramid Network (FPN) for multi-scale feature extraction.
	•	Attention-Enhanced Mask R-CNN: Added SEBlock attention mechanisms to further refine feature learning within the network.

Implementation Details
	•	Models were implemented in PyTorch and trained on NVIDIA RTX 4070 and 4090 GPUs.
	•	Optimizers such as AdamW (with learning rate scheduling) and SGD were employed to enhance training stability.
	•	Extensive evaluation metrics included mean Intersection over Union (mIoU) for key regions (e.g., turtle body, flippers, head), with additional assessments of computational efficiency.

## Results

•	U-Net with CBAM achieved mIoU: 0.8316.

•	Mask R-CNN (ResNet-50 + FPN) achieved the best mIoU: 0.9117.

Both models demonstrated unique strengths, with U-Net excelling in detail-oriented tasks and Mask R-CNN proving superior for complex, instance-level segmentation in noisy underwater environments.

Detailed results are documented in the report.

## Directory Contents
```
├── UNet_final.ipynb                    
├── Mrcnn_with_ResNet50_final.ipynb     
├── Mrcnn_with_attention_final.ipynb   
├── Dataset Exploration.html           
├── requirements.txt                  
├── report.pdf                        
└── README.md                         
```

•   **UNet_final.ipynb**: This notebook contains our implementation of the U-Net model for segmentation. It includes parameter settings, optimizations, and final results.

•   **Mrcnn_with_ResNet50_final.ipynb**: This notebook provides the code for our Mask R-CNN model using a ResNet50 backbone, set up for sea turtle segmentation. It includes parameter settings, optimizations, and final results.

•   **Mrcnn_with_attention_final.ipynb**: This notebook enhances the Mask R-CNN model by integrating attention mechanisms for improved segmentation performance. It includes parameter settings, optimizations, and final results. (developed in Google Colab)

•   **Dataset Exploration.html**: This is the HTML export of the dataset exploration and mask visualization notebook.

•   **requirements.txt**: A list of dependencies needed to run the notebooks. Please follow the usage instructions below.

•   **report.pdf**: This file contains the detailed report of the project, including methodology, results, and analysis.

•   **README.md**: This documentation file.


## Usage Instructions


1. Setting up the Local Environment:
\
These notebooks are compatible with **Python 3.9** and **Python 3.10**. Using one of these versions ensures compatibility with all dependencies listed in requirements.txt and provides optimal stability and performance.

To check your Python version, use:
```sh
python --version
```

**Recommendation**: Install the dependencies listed in requirements.txt.

```sh
pip install -r requirements.txt
```

2.	Running the Notebooks:
\
Open `UNet_final.ipynb` or `Mrcnn_with_ResNet50_final.ipynb` in a compatible IDE and Mrcnn_with_attention_final.ipynb in Google Colab. 

**Note**: `Mrcnn_with_attention_final.ipynb` is developed in Google Colab, so please make sure to upload the necessary files to the Colab environment or change the file paths accordingly if you want to run it in your local environment.

Execute the cells sequentially to reproduce the results.

**Note**: These notebooks only provide final experiment findings for results replication. Please refer to the report for further details on experimental setups, results, and analysis.

## Acknowledgements

We would like to thank our instructors and peers for their support and guidance throughout this project.


**Contributors**

• Chenhao Su, Email: z5503282@ad.unsw.edu.au

• Jo Jin, Email: z5510401@ad.unsw.edu.au

• Ivan Luk, Email: z5463348@ad.unsw.edu.au

• Xuedong Zhang, Email: z5582401@ad.unsw.edu.au

• Yi Han, Email: z53652201@ad.unsw.edu.au

## License
This project is licensed under the MIT License - see the LICENSE file for details.
