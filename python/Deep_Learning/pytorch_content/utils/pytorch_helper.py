import torch
import torch.nn as nn

# For Type Annotations
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
from torch import device
from torch.nn import Module

def check_gpu_avaible():
    """Checking the gpu status and returns the torch device if a gpu on
    mac or cuda is avaible.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Running on MPS. M1 GPU is avaible")
        
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on Nvidia. Cuda is avaible")
        
    else:
        device = torch.device("cpu")
        print("Running on CPU")
        
    return device


def train_model_with_Keypoints(model: Module,
                                n_epochs: int,
                                train_loader: DataLoader,
                                valid_loader: DataLoader,
                                device: device,
                                optimizer: Optimizer,
                                criterion: _Loss,
                                clip_grad: bool=True, 
                                best_model_score: float=10.0*10,
                                save_path: str=None):
    """_summary_

    Args:
        model (Module): _description_
        n_epochs (int): _description_
        train_loader (DataLoader): _description_
        valid_loader (DataLoader): _description_
        device (device): _description_
        optimizer (Optimizer): _description_
        criterion (_Loss): _description_
        clip_grad (_type_, optional): _description_. Defaults to True:bool.
        best_model_score (_type_, optional): _description_. Defaults to None:float.
        save_path (_type_, optional): _description_. Defaults to None:str.

    Returns:
        _type_: _description_
    """
    
    model.to(device)
    model.train()
    
    print_every_n = 10
    best_model_dict = model.state_dict()
    
    print("Starting  Model training")
    print("_"*50 + "\n")
    
    for epoch in range(n_epochs):
        
        batch_loss = 0
        train_epoch_loss = 0
        
        for batch_i, data in enumerate(train_loader, 1):
            # get the input images and their corresponding labels
            images = data["image"]
            key_pts = data["keypoints"]
            
             # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)
            
            # forward pass to get outputs
            output_pts = model(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # clipping gradients
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            # update the weights    
            optimizer.step()
            
            # Calculalating batch and epoch loss (training) - 
            # by default the loss is averaged, so mutiply it by the batch size
            batch_loss += loss.item() * images.size(0)
            train_epoch_loss += loss.item() * images.size(0)
            
            
            if batch_i % print_every_n == 0:    # print every n batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch+1,
                                                                   batch_i,
                                                                   batch_loss/print_every_n))
                
                print(40*"-")
                
                # reset batch loss
                batch_loss = 0.0
                
                # Evaluation of the trained model
                model.eval()
                valid_score = 0
                # valid model on validation loader
                with torch.no_grad():
                    for data in valid_loader:
                        
                        images = data["image"]
                        key_pts = data["keypoints"]
                        
                        # flatten pts 
                        key_pts = key_pts.view(key_pts.size(0), -1)

                        # convert variables to floats for regression loss
                        key_pts = key_pts.type(torch.FloatTensor).to(device)
                        images = images.type(torch.FloatTensor).to(device)
                        
                        # forward pass to get outputs
                        output_pts = model(images)

                        # calculate the loss between predicted and target keypoints
                        loss = criterion(output_pts, key_pts)
                        
                        # adding loss to total loss
                        valid_score += loss.item() * images.size(0)
                        
                    total_valid_mse = valid_score / len(valid_loader)
                    
                    if total_valid_mse < best_model_score:
                        print("\n" + "*"*50)
                        print("Total MSE in validation set decreased")
                        print(f"New: {total_valid_mse} - Old: {best_model_score}")
                        print(f"Model state saved at - Epoch: {epoch+1} - Batch: {batch_i}")
                        print("*"*50 + "\n")
                        best_model_score = total_valid_mse
                        best_model_dict = deepcopy(model.state_dict())
                        
                        if save_path is not None:
                            model_checkpoint = {"epoch": epoch+1,
                                                "batch": batch_i,
                                                "valid_loss": best_model_score,
                                                "model_state_dict": best_model_dict,
                                                "optimizer": optimizer.state_dict(),
                                                "loss_fn": criterion.__str__()}
                            
                            torch.save(model_checkpoint, save_path)
                        
                model.train()             
                          
        # print epoch loss
        train_epoch_loss = train_epoch_loss / len(train_loader)        
        print(f"Total Training Loss Epoch {epoch+1}: {train_epoch_loss}")
        print("_"*20)
    print('Finished Training')
    
    return best_model_dict, best_model_score