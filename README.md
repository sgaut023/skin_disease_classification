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
![samples](https://user-images.githubusercontent.com/23482039/145730125-68d86857-7caf-40f1-92d7-8b5e3696ddce.png)
The task consists of classifying three skin disease types from skin images. The three classes are “acne”, “herpes simplex”, and “lichen planus”.  In total, there are 102 images: 40 images of acne, 16 images of herpes simplex and 46 of lichen planus.

There are three key challenges associated with this dataset. 

Challenge 1: **Learning from an imbalance dataset**. The dataset is imbalanced considering the fact that there are only 16 images of herpes simplex. We need to take this information into account during the training of the different models. 

Challenge 2: **Efficient learning from limited labeled data**. One of the key challenges here from a machine-learning perspective is a limited amount of labeled data. Learning useful representations from little training data is arduous. The model can easily overfit the small training set. 

Challenge 3: **Finding disentangled representations**. A crucial element in extracting information features from high dimensional structured data is the disentanglement of sources of variations in the data. We observe a lot of intra-class variation. Let’s visualize some of the images.

Architectures
------------
We consider four different architectures:
 - Vanilla CNN
 - Pretrained Resnet-50
 - [BiT](https://arxiv.org/abs/1912.11370)
 - [Parametric Scattering Network](https://arxiv.org/abs/2107.09539)

In notebook [2.0.HyperParameterSearch.ipynb](https://github.com/sgaut023/skin_disease_classification/blob/main/2.0.HyperParameterSearch.ipynb), we train the different models using cross-validation. We perform a simple hyperparameter search. The model that yields the highest macro average recall over the 5 folds is the [BiT](https://arxiv.org/abs/1912.11370). 

Training and Saving Model
------------
In notebook [3.0.TrainEval.ipynb (https://github.com/sgaut023/skin_disease_classification/blob/main/3.0.TrainEval.ipynb), we train the final model using the best combination of hyperparameters found in the second notebook. The state dictionary of the final [BiT](https://arxiv.org/abs/1912.11370) model is saved [here](https://github.com/sgaut023/skin_disease_classification/tree/main/trained_models).

Inference
------------
If you want to use the trained model to infer on unseen images, please run notebook [34.0.Inference.ipynb](https://github.com/sgaut023/skin_disease_classification/blob/main/4.0.Inference.ipynb). The new images need to be in the [random_images/](https://github.com/sgaut023/skin_disease_classification/tree/main/random_images) folder. The predictions are saved in the predictions.csv file. 

Experimental Tracking Tool
------
All the results of notebook [2.0.HyperParameterSearch.ipynb](https://github.com/sgaut023/skin_disease_classification/blob/main/2.0.HyperParameterSearch.ipynb) have been saved using [MLflow](https://mlflow.org/) tracking tool. The results are saved in the the DAGsHub's servers. Click [here](https://dagshub.com/gauthier.shanel/skin_disease/experiments/#/) to see the hyperparameter search results. 

![image](https://user-images.githubusercontent.com/23482039/145730274-48763c77-9225-486b-b08e-b4300df04564.png)


Project Organization
------------

    ├── report.pdf                     <- final detailed report 
    ├── utils.py                       <- utility functions
    ├── models.py                      <- pytorch nn.Module (vanilla CNN and pretrained Resnet-50)
    ├── environment.yml                <- The conda environment file for reproducing the analysis environment.
    ├── data_3class_skin_diseases.zip  <- dataset
    ├── 1.0.Data-Exploration.ipynb     <- Notebook exploring dataset
    ├── 2.0.HyperParameterSearch.ipynb <- Notebook evaluating performance of 4 architectures (vanilla cnn, pretrained resnet-50, bit and parametric scattering network)
    ├── 3.0.TrainEval.ipynb            <- Notebook training and saving final model (big transfer model)
    ├── 4.0.Inference.ipynb            <- Notebook doing inference on unseen images
    ├── trained_models   
    │   ├── bit.pth                    <- state dictionary of final bit model
    ├── class_names.json               <- mapping between label numbers and label names
    ├── predictions.csv                <- saved predictions of notebook 4.0 (inference notebook)
    ├── random_images                  <- dummy dataset created from the Internet (to do inference)
    

