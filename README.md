# Skin Disease Classification

Get Setup
------------

Start by cloning the repository
```
git clone https://github.com/sgaut023/skin_disease_classification.git
```

Conda
------
Prerequisites
- Anaconda/Miniconda 

To create the `oro` conda environment, enter the following in the command prompt: 
```
conda env create -f environment.yml
```
To active the `oro_health` conda environment, enter the following: 
```
conda activate oro_health
```
Datasets
------------
The task consists of classifying three skin disease types from skin images. The three classes are “acne”, “herpes simplex”, and “lichen planus”.  In total, there are 102 images: 40 images of acne, 16 images of herpes simplex and 46 of lichen planus.

There are three key challenges associated with this dataset. 


Challenge 1: **Learning from an imbalance dataset**. The dataset is imbalanced considering the fact that there are only 16 images of herpes simplex. We need to take this information into account during the training of the different models. 

Challenge 2: **Efficient learning from limited labeled data**. One of the key challenges here from a machine-learning perspective is a limited amount of labeled data. Learning useful representations from little training data is arduous. The model can easily overfit the small training set. 

Challenge 3: **Finding disentangled representations**. A crucial element in extracting information features from high dimensional structured data is the disentanglement of sources of variations in the data. We observe a lot of intra-class variation. Let’s visualize some of the images.

