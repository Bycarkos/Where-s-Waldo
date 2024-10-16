
## Dataset Things
from data.volumes import Volume, Page, Line
from data.graphset import Graphset
from data.graph_sampler import AttributeSampler
from data.ced_dataset import CEDDataset
from data.esposalles_dataset import EsposallesDataset
from data.iam_dataset import IAMDataset
from data.washington_dataset import WashingtonDataset
import data.volumes as dv

## Model Things
from models import visual_encoders as VE
import pytorch_warmup as warmup
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from losses.perceptual_loss import ResNetFeatureExtractor, VGGFeatureExtractor

### Utils
import utils 
import visualizations as visu

## Pipelines
import pipelines as pipes


## Common packages
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam


## Typing Packages
from typing import *

## Configuration Package
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra import initialize, compose

## Experiment Tracking packages
import wandb
import tqdm


## Common packages
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from beeprint import pp as bprint
import pdb
from math import ceil


device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Red (!)
Blue (?)
Green (*)
Yellow (^)
Pink (&)
Purple (~)
Mustard (todo)
Grey (//)
Commented out Code can also be Styled to make it Clear the Code shouldn't be There.
Any Other Comment Styles you'd like can be Specified in the Settings.
"""
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        
def batch_step(loader: Type[DataLoader],
               model: Type[nn.Module],
               criterion: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               scheduler,
               sobel_kernel: Optional[torch.Tensor]=None,
               epoch:int=0):
    
    model.train()
    perceptual_loss = ResNetFeatureExtractor().to(device)
    if sobel_kernel is not None:
        x_kernel, y_kernel = sobel_kernel
    batch_loss = 0

    for idx, batch in tqdm.tqdm(enumerate(loader), desc="Batch Loop", leave=True, position=1):
        optimizer.zero_grad()
        images = batch["image_lines"].to(device)
        original_images = batch["non_augmented_image_lines"].to(device)
        masks = batch["masks"].to(device)

        #visu.plot(original_images.cpu())
        #exit()

        ocr = batch["ocrs"]

        reconstructed_image = model(images)

        original_images = torch.clip(original_images * STD + MEAN, min=0., max=1.)
        #reconstructed_image = reconstructed_image * STD + MEAN

        MSE = criterion(original_images, reconstructed_image)
        
        loss = (torch.mean(MSE)) + (torch.mean(MSE * masks)*10) + perceptual_loss(reconstructed_image, original_images) * 0.7
        #loss = torch.mean(MSE)
        #loss += perceptual_loss(reconstructed_image, original_images)
      

        batch_loss += loss

        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_loss = batch_loss/len(loader)

    wandb.log({"Training Loss": epoch_loss}, step=epoch)

    return epoch_loss

@torch.no_grad()
def eval(loader: Type[DataLoader],
               model: Type[nn.Module],
               criterion: Type[nn.Module],
               mode: str="Validation",
               epoch:int = 0):
    model.eval()

    batch_loss = 0

    original_images_grid = None
    reconstructred_images_grid = None
    masks_grid = None

    perceptual_loss = ResNetFeatureExtractor().to(device)

    for idx, batch in tqdm.tqdm(enumerate(loader), desc="Validation/Test Batch Loop", leave=True, position=1):
        images = batch["image_lines"].to(device)
        original_images = batch["non_augmented_image_lines"].to(device)
        ocr = batch["ocrs"]
        masks = batch["masks"].to(device)

        reconstructed_image = model(images)

        original_images = torch.clip(original_images * STD + MEAN, min=0., max=1.)
        #reconstructed_image = reconstructed_image * STD + MEAN

        #visu.plot([original_images[0].detach().cpu(), reconstructed_image[0].detach().cpu()])
        #exit()
        if idx ==0:
            original_images_grid = original_images
            reconstructred_images_grid = reconstructed_image
            masks_grid = masks

        else:
            original_images_grid = torch.vstack((original_images_grid, original_images))
            reconstructred_images_grid = torch.vstack((reconstructred_images_grid, reconstructed_image))
            masks_grid = torch.vstack((masks_grid, masks))


        MSE = criterion(original_images, reconstructed_image)

        loss = (torch.mean(MSE)) + (torch.mean(MSE * masks)*10) + perceptual_loss(reconstructed_image, original_images) * 0.7

        batch_loss += loss


    original_images = wandb.Image(torchvision.utils.make_grid(original_images_grid[:100], nrow=16), caption="Original")
    reconstructed = wandb.Image(torchvision.utils.make_grid(reconstructred_images_grid[:100], nrow=16), caption="Reconstructed")
    masks_grid = wandb.Image(torchvision.utils.make_grid(masks_grid[:100], nrow=16), caption="Masks")

    
    wandb.log({f"{mode}: AE evaluation Original Images":original_images}, step=epoch)
    wandb.log({f"{mode}: AE evaluation Reconstructed images":reconstructed}, step=epoch)
    wandb.log({f"{mode}: AE evaluation Masks":masks_grid}, step=epoch)

    epoch_loss = batch_loss/len(loader)

    wandb.log({"Validation Loss": epoch_loss}, step=epoch)

    return epoch_loss


@hydra.main(config_path="./configs", config_name="pretrain", version_base="1.3.2")
def main(cfg: DictConfig):
    CFG_DATA = cfg.data
    CFG_MODELS = cfg.models
    CFG_SETUP = cfg.setup
    
    if cfg.log_wandb == True:

        if cfg.verbose == True :
            print("Wandb configuration: ")

            bprint(dict(CFG_SETUP.wandb))

        wandb.login(key="ab18aafa8c70616ba4ef66844fc9444794cae54a", relogin=True)
        wandb.init(
            project= CFG_SETUP.wandb.project,
            config = dict(CFG_SETUP.wandb.config),
            name = CFG_SETUP.wandb.name,
            group = CFG_SETUP.wandb.group
        )

    #  ^ Hydra things
    epochs = CFG_SETUP.configuration.epochs
    batch_size = CFG_DATA.collator.batch_size
    shuffle = CFG_DATA.collator.shuffle
    checkpoint_name = CFG_MODELS.name_checkpoint
    partitions_ratio = CFG_DATA.collator.partitions_ratio
    # ^ 

    # *Extracting the Datasets
    means_width = []
    means_heigth = []

    dict_datasets = {}
    
    for dataset_object in (CFG_DATA.datasets):
        data = instantiate(CFG_DATA.datasets[dataset_object])
        dict_datasets[dataset_object] = data

        means_heigth.append(np.mean(data.line_heights))
        means_width.append(np.mean(data.line_widths))

    standarized_width = ceil(np.mean(means_width))
    standarized_height = ceil(np.mean(means_heigth))

    for dts_name, dts  in dict_datasets.items():
        dts.define_transforms((standarized_height, standarized_width))
    
    list_of_datasets = list(dict_datasets.values())
    if len(list_of_datasets) == 1:
        print("Only training with one Dataset")
        merged_dataset = list_of_datasets[-1]
    else:
        merged_dataset = torch.utils.data.ConcatDataset(list_of_datasets)
    
    collate_fn = list_of_datasets[0].collate_fn

    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(merged_dataset, partitions_ratio, generator=generator)
    # *


    # * Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=8, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=1, collate_fn=collate_fn)
    # *
    

    print("DATA LOADER SUCCESFULLY GENERATED")

    model = VE.LineAutoEncoder(cfg=CFG_MODELS.visual_encoder).to(device)

    if CFG_MODELS.finetune is True:
        model_name = f"./checkpoints/{checkpoint_name}.pt"
        
        model.load_state_dict(torch.load(model_name))
        print("Model Loaded Succesfully Starting Finetunning")


    optimizer = hydra.utils.instantiate(CFG_SETUP.optimizer, params=model.parameters())
    criterion = nn.MSELoss(reduction="none")
    
    
    ## ** Scheduler
    num_warmup_steps = 1000  
    train_loader_len = len(train_loader)  # Assuming you have defined train_loader
    total_steps = epochs * train_loader_len
    num_cycles = epochs // 2
    
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        num_cycles=num_cycles,
    )

    ## **
    optimal_loss = 10000

    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{checkpoint_name}.pt"


    ## Start the training
    print("CREATING THE BASELINE METRIC VALUE\n STARTING TO EVALUATE FO THE FIRST TIME")
    loss_validation = eval(loader=validation_loader,
                           model=model,
                           criterion=criterion)
    


    _, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                actual_metric=loss_validation, 
                                model=model, 
                                checkpoint_path=checkpoint_name, 
                                compare="<")

    print(f"Validation Loss Epoch: {0} Value: {loss_validation} Optimal_loss: {optimal_loss}")
    
    x_kernel, y_kernel = utils.get_sobel_kernel(device=device, chnls=3)
    
    for epoch in tqdm.tqdm(range(epochs), desc="Training Process", position=0, leave=False):
        
    
        train_loss = batch_step(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            scheduler=scheduler
        ) ## to fulfill


            #wandb.log({"Validation Loss": loss_validation})
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # Get current learning rate (from scheduler)
        print(f"Loss Epoch: {epoch} Value: {train_loss} LR: {current_lr:.6f}")

        if ((epoch +1) % 10) == 0:
            loss_validation = eval(loader=validation_loader,
                                model=model,
                                criterion=criterion,
                                epoch=epoch)

            updated, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                        actual_metric=loss_validation, 
                                        model=model, 
                                        checkpoint_path=checkpoint_name, 
                                        compare="<")
            
            if updated:
                print(f"Model Updated: Validation Loss Epoch: {0} Value: {loss_validation} Optimal_loss: {optimal_loss}")
            



                    
    model.load_state_dict(torch.load(checkpoint_name))

    loss_test = eval(loader=test_loader,
                     model=model,
                     criterion=criterion,
                     mode="Test",
                     epoch=epoch)
    
    updated, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                actual_metric=loss_test, 
                                model=model, 
                                checkpoint_path=checkpoint_name, 
                                compare="<")
    
    if updated:
        print(f"Model Updated on Test")



if __name__ == "__main__":
    main()