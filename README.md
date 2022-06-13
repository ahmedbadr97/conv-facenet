# conv-facenet
- using conv-next as base model for face recognition with pytorch
- the model trained on CelebFaces large-scale face dataset with more than 200K celebrity images with accuracy about 93.8%

# Dataset
## [CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 
<p align="center"><img src="readme-assets/Celba.png" alt="img-celba" width="720" height="250" /> </p>  

- CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images
- The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities
  - 10,177 number of identities
  - 202,599 number of face images
  - dataset is about 10 GB
- ### sample images
  - <img src="readme-assets/sample-dataset.png" alt="img-celba" width="720" height="300" />
# Data preprocessing and preparation
- [Data preprocessing jupyter file](notebooks/data_preprocessing.ipynb)
- using face detector [Pytorch-Retinaface](https://github.com/biubug6/Pytorch_Retinaface) from [biubug](https://github.com/biubug6) to detect and align faces to get better learning performance by getting the left and right eye coordinates and rotate the image to make the two eyes on same horizontal line and make them at the center of the image
- resize photos after face extraction and alignment to be square photos 240x240 as input size for the model without image deformation by filling the remaining pixels with black pixels to make the image square image then resize it
- after image resize and face extraction the size of the data decreased from around 10 GB to 2.82 GB which fastens the training performance
- 




