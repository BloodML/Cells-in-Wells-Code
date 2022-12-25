# Cells-in-Wells-Code

## Introduction
This repository contains the code and trained neural networks that were used to create the ["Cells-in-Wells" (CIW) dataset](https://doi.org/10.18738/T8/PSQKWF), which is composed of images, videos, binary masks, bounding boxes, morphology labels, cell tracking data, and other metrics of red blood cells (RBCs) in microfluidic wells undergoing washing in human serum albumin, saline, or autologous storage medium.  
![RBCs in wells](https://github.com/BloodML/Cells-in-Wells-Code/blob/main/Donor%204/HSA/1/image_000001.png?raw=true)
What follows is a conceptual overview and a brief tutorial on how to use the code.

## Conceptual Overview
![The pipeline.](https://raw.githubusercontent.com/BloodML/Cells-in-Wells-Code/main/WorkflowDiagram.svg)



A conglomeration of MATLAB examples, tutorials, and research articles inspired the pipeline’s segmentation, deep learning, and object tracking architecture (Eddins & MathWorks, 2002; Howell et al., 2020; Kenta, 2022; MathWorks, 2022a, 2022b, 2022c, 2022d). First, 1280X1024 grayscale images are put through preprocessing, which consists of segmentation and classification. Segmentation occurs via image analysis or a combination of image analysis and semantic segmentation. Invariably, this step aims to obtain image masks, blob analysis statistics, and RBC bounding boxes, which enable downstream analysis and processing. During classification, each bounding box cropped RBC image is normalized and shown to an ensemble of four CNN classifiers. Labels correspond to the maximum value of the averaged softmax scores, which is one of seven possibilities (ST = Stomatocyte, D=discocyte, E1,2,3= echinocyte 1,2,3, SE = Sphero-echinocytes, and S = spherocyte). In the pipeline’s second stage, all the outputs from the preprocessing stage are used with a data structure that tracks each cell through a Kalman filter predicting the RBC’s future position. Tracking allows us to average the softmax scores for cells over multiple frames, which can be seen as a simple extension of CNNs to analyze time-series data. Finally, through data analysis of the cell records, we can acquire biologically relevant information, such as morphological indices and statistical relationships. Furthermore, an expert can use the outputs to visually inspect each well or improve the system by manually curating the binary masks and labeling data for feedback.

## Requirements
- MATLAB 2021b or higher
- Deep Learning Toolbox (14.3)
- Image Processing Toolbox (11.4) 
- Statistics and Machine Learning Toolbox (12.2)
- Computer Vision Toolbox (10.1)

## Tutorial
### Test Run
To test your MATLAB setup and the code. 
1. Download and extract the repository.
2. Delete the **Preprocessed Data** & **Processed Data** folders
	- The code comes with data from the HSA wash of Donor 4 (**Donor 4** folder), which we will use to test the framework by regenerating the preprocessed & processed data.
3. Open the **Routt_Austin_CIW_Preprocessing_Main.m** script in MATLAB and press **Play**
	- When finished, the **Preprocessed Data**  folder and all of its contents are recreated.
4. Open the **Routt_Austin_CIW_Processing_Main.m** script in MATLAB and press **Play**
	- When finished, the **Processed Data**  folder and all of its contents are recreated.
5. If both scripts execute without error and both folders are recreated, your setup and the code are in working order.

### Reproduce the CIW Dataset from the raw microscope data
The CIW dataset has the donor data in multivolume zip files, where Donor # #####.zip is the head (e.g. **Donor 6 12917.zip**) and the multiple Donor # #####.z## files correspond to the body of each donor (e.g. **Donor 6 12917.z01** to **Donor 6 12917.z18**). You need to download the head and the body files for all 6 donors to fully reproduce the CIW dataset.
1. Go to the [CIW dataverse](https://doi.org/10.18738/T8/PSQKWF) repository to download, and then extract all 6 donors. 
	- Only extract the images for each donor wash because we are reproducing the data via the code. 
2. After extracting the images into their respective donor-wash folders (e.g. **Donor 1 11617\HSA\1\Images**) go into the **Routt_Austin_CIW_Preprocessing_Main.m** script, set the base import & export paths on lines  67 & 68, and uncomment lines 76 & 77. 
3. Press **Play** and let the script collect the data into the output folder. where the **Preprocessed Data** folder is the default.
4.  Watch the MATLAB console display the percentage of completion until the script finishes.
5. Repeat this process using the **Routt_Austin_CIW_Processing_Main.m** script and the **Preprocessed Data**. 
	- Open the script, uncomment lines 19 & 20, and press **play**.
6. Check the **Processed Data** folder for individual RBC tracking data and well videos.
![example 1](https://github.com/BloodML/Cells-in-Wells-Code/blob/main/SupFig15.gif?raw=true)
![example 2](https://github.com/BloodML/Cells-in-Wells-Code/blob/main/SupFig16.gif?raw=true)

![example 3](https://github.com/BloodML/Cells-in-Wells-Code/blob/main/SupFig17.gif?raw=true)

## References

Eddins, S., & MathWorks. (2002). _The Watershed Transform: Strategies for Image Segmentation_. [https://www.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html](https://www.mathworks.com/company/newsletters/articles/the-watershed-transform-strategies-for-image-segmentation.html)

 Howell, J., Hammarton, T. C., Altmann, Y., & Jimenez, M. (2020). High-speed particle detection and tracking in microfluidic devices using event-based sensing. _Lab Chip_, _20_(16), 3024-3035. [https://doi.org/10.1039/d0lc00556h](https://doi.org/10.1039/d0lc00556h)

Kenta. (2022). _Oversampling for deep learning: classification example_. [https://github.com/KentaItakura/Image-classification-using-oversampling-imagedatastore/releases/tag/2.0](https://github.com/KentaItakura/Image-classification-using-oversampling-imagedatastore/releases/tag/2.0)

MathWorks. (2022a). _assignDetectionsToTracks_. [https://www.mathworks.com/help/vision/ref/assigndetectionstotracks.html](https://www.mathworks.com/help/vision/ref/assigndetectionstotracks.html)

MathWorks. (2022b). _Semantic Segmentation With Deep Learning_. [https://www.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html](https://www.mathworks.com/help/vision/ug/semantic-segmentation-with-deep-learning.html)

MathWorks. (2022c). _Train Deep Learning Network to Classify New Images_. [https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html](https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html)

MathWorks. (2022d). _Use Kalman Filter for Object Tracking_. [https://www.mathworks.com/help/vision/ug/using-kalman-filter-for-object-tracking.html](https://www.mathworks.com/help/vision/ug/using-kalman-filter-for-object-tracking.html)

