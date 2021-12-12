import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from models import Net, Resnet50

import time
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sn
import mlflow
import matplotlib.pyplot as plt
import sklearn.utils.class_weight as class_weight
import copy
import timm

from kymatio_mod.parametricSN.models.sn_base_models import sn_ScatteringBase
from kymatio_mod.parametricSN.models.sn_top_models import sn_LinearLayer
from kymatio_mod.parametricSN.models.sn_hybrid_models import sn_HybridModel

def get_loaders(dataset, train_ids, test_ids, batch_size):
    """
    Creates train loader and test loader.

    Args:
        dataset (torchvision.datasets.folder.ImageFolder): generated dataset
        train_ids (list): list of dataset training set indexes
        test_ids (list): list of dataset test set indexes
        batch_size (int): mini batch size

    Returns:
        trainloader (torch.utils.data.DataLoader): train dataloader
        testloader (torch.utils.data.DataLoader):  test dataloader
        train_subsampler (torch.utils.data.SubsetRandomSampler): train subsampler
    """  
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader( dataset,batch_size=batch_size, sampler=test_subsampler)
    return trainloader, testloader, train_subsampler

def get_loss_function(dataset, trainloader, indices, device=None):
    """
    Since the dataset is imbalanced, the loss function is weighted.
    This function creates the weighted loss function by computing the class weight.

    Args:
        dataset (torchvision.datasets.folder.ImageFolder): generated dataset
        trainloader (torch.utils.data.DataLoader): train dataloader
        indices (list): list of dataset training set indexes

    Returns:
        loss (nn.CrossEntropyLoss): loss function
    """  
    #Compute class weight of the classes
    class_weights=class_weight.compute_class_weight('balanced', classes= np.unique(dataset.targets), 
                                                    y = np.array(trainloader.dataset.targets)[indices] )
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    return nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

def get_confusion_matrix(y_true, y_pred):
    """
    Visualizes confusion matrix.

    Args:
        y_true (list): list of labels
        y_pred (list): list of predicted values
    """    
    matrix = confusion_matrix(y_true, y_pred)
    index = np.arange(0, len(matrix), 1)
    df_cm = pd.DataFrame(matrix, index=index, columns=index)
    f = plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def visualize_loss(train_loss, test_loss):
    """
    Visualizes train and test losses over epochs. 

    Args:
        train_loss (list): list of train losses 
        test_loss (list): list of test losses 
    """ 
    f = plt.figure(figsize=(7,5))
    epochs = np.arange(0, len(train_loss))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot( epochs, train_loss, label='Train' )
    plt.plot( epochs, test_loss, label= 'Test' )
    plt.legend()
    plt.show()

def evaluate_model(network, testloader, loss_function, device):
    """
    Evaluates trained model using test set

    Args:
        network (nn.Module): model
        testloader (torch.utils.data.DataLoader): test dataloader
        loss_function (torch.nn): loss function

    Returns:
        loss (float): test loss
        y_pred (list): list of predicted values
        y_true (list): list of labels

    """   
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = []
        y_true = []
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = network(inputs)
            loss = loss_function(outputs, targets)  
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    return loss, y_pred, y_true

def save_results(results, fold, y_pred, y_true):  
    """
    Saves results in 'results' dictionary
    Args:
        results (dict): dictionary containing all results for every fold
        fold (int): fold number
        y_pred (list): list of predicted values
        y_true (list): list of true labels
    """   
    results[fold]={}
    results[fold]['pred'] = y_pred
    results[fold]['label']= y_true
    results[fold]['classification_report']=classification_report(y_true, y_pred, output_dict=True)

def setAllSeeds(seed):
    """Helper for setting seeds"

    Args:
        seed (int): seed number
    """   
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train_model(name, dataset, k_folds =5, num_epochs =15, lr =1e-4, random_state=42, batch_size=16, device=None):
    """Trains model
       Code isnpired from: https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch/
    Args:
        name (str): architecture type ('cnn', 'resnet50', 'bit', 'scattering')
        dataset  (torchvision.datasets.folder.ImageFolder): generated dataset
        k_folds (int, optional): number of folds. Defaults to 5.
        num_epochs (int, optional): number of epochs. Defaults to 15.
        lr ([type], optional): learning rate. Defaults to 1e-4.
        random_state (int, optional): seed number. Defaults to 42.
        batch_size (int, optional): batch size. Defaults to 16.

    Returns:
        results (dict): dictionary of results
    """  

    setAllSeeds(random_state)
    kfold = StratifiedKFold(n_splits=k_folds, shuffle = True, random_state=random_state)
    results = {}
    start_time = time.time()
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset, dataset.targets)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        network = get_model(name)
        network.to(device)
        trainloader, testloader, train_subsampler =get_loaders(dataset, train_ids, test_ids, batch_size)
        loss_function = get_loss_function(dataset, trainloader, train_subsampler.indices, device=device)
        optimizer = torch.optim.SGD(network.parameters(), lr=lr)

        # Run the training loop for defined number of epochs
        train_loss = []
        test_loss= []
        for epoch in range(0, num_epochs):
            current_loss = 0.0
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                outputs = network(inputs)
                loss = loss_function(outputs, targets)  
                loss.backward()   
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
        train_loss.append(current_loss/(i+1))
        loss, y_pred, y_true = evaluate_model(network,testloader, loss_function)
        test_loss.append(loss.item())
        print(f'Epoch {epoch+1}-- Train Loss:{current_loss/(i+1)}, Test Loss: {loss}')

        visualize_loss(train_loss, test_loss) 
        get_confusion_matrix(y_true, y_pred)
        print('Starting testing')
        loss, y_pred, y_true = evaluate_model(network,testloader, loss_function)
        save_results(results, fold, y_pred, y_true)

    total_time = time.time() - start_time
    avg_time_fold = total_time/k_folds
    results['total_time'] =total_time
    results['avg_time_fold']= avg_time_fold
    results['trainable_parameters'] = sum(p.numel() for p in network.parameters() if p.requires_grad)
    results['batch_size']=batch_size
    results['lr'] =lr
    results['num_epochs']= num_epochs
    print('Training Complete')
    return results

def get_avg_classification_report(results, k_folds=5):
    """
    Creates final classification report by averaging over all fold classification reports 

    Args:
        results (dict): dictionary of results
        k_folds (int, optional): number of folds. Defaults to 5.

    Returns:
        avg_report (dict): final classification report (average over all fold reports)
    """ 
    avg_report = copy.deepcopy(results[0]['classification_report'])
    for key in avg_report:
        for sub_key in ['precision', 'recall', 'f1-score']:
            values = []
            is_subkey = True
            for fold in np.arange(0,k_folds):
                try:
                    values.append(results[fold]['classification_report'][key][sub_key])
                except:
                    values.append(results[fold]['classification_report'][key])
                    is_subkey = False
                mean = np.array(values).mean()
                std = np.array(values).std()
                if is_subkey:
                    avg_report[key][sub_key] = f'{mean.round(3)}±{std.round(3)}'
                else:
                    avg_report[key] = f'{mean.round(3)}±{std.round(3)}'
    return avg_report

def save_metrics_mlflow(name, report, results ):
    """Logs all metrics in MLFLOW

    Args:
        name (str): architecture type ('cnn', 'resnet50', 'bit', 'scattering')
        report (dict): final average classification report (average of all fold reports)
        results (dict): dictionnary of all results
    """   
    with mlflow.start_run(run_name = 'skin_disease_performance'):
        mlflow.log_param('model', name)
        mlflow.log_param('total_time',results['total_time'] )
        mlflow.log_param('avg_time_fold',results['avg_time_fold'] )
        mlflow.log_param('lr', results['lr'])
        mlflow.log_param('batch_size', results['batch_size'])
        mlflow.log_param('num_epochs', results['num_epochs'])
        mlflow.log_param('trainable_parameters', results['trainable_parameters'])
        for key in report:
            for sub_key in ['precision', 'recall', 'f1-score']:
                try:
                    values = report[key][sub_key].split('±')
                    mlflow.log_metric(f'{key}-{sub_key}-avg',float(values[0]))
                    mlflow.log_metric(f'{key}-{sub_key}-std',float(values[1]))
                except:
                    values = report[key].split('±')
                    mlflow.log_metric(f'{key}-avg', float(values[0]))
                    mlflow.log_metric(f'{key}-std', float(values[1]))

def get_model(name):
    """
     Created and returns desired model

    Args:
        name (str): architecture type ('cnn', 'resnet50', 'bit', 'scattering')

    Returns:
        network (nn.Module): model
    """    
    if name =='cnn':
        network = Net()
    elif name =='resnet50':
        network =  Resnet50(num_classes=3)
    elif name =='bit':
        network = timm.create_model('resnetv2_101x1_bitm', pretrained=True, num_classes=3)
    elif name=='scattering':
        scatteringBase = sn_ScatteringBase(J=2, N=400, M=400, second_order=False, initialization='Tight-Frame', seed=42, 
                                        learnable=True, lr_orientation=0.1, lr_scattering=0.1, monitor_filters=True,
                                        filter_video=False, parameterization='canonical')
        plt.close('all')
        ll_net = sn_LinearLayer(num_classes=3, n_coefficients=81, M_coefficient=100, N_coefficient=100)
        network = sn_HybridModel(scatteringBase=scatteringBase, top=ll_net ) 
    else: 
        NotImplemented(f"Model Architecture {name} not implemented")
    return network




