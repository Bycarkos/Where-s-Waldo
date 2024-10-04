
## Dataset Things
from data.volumes import Volume, Page, Line
from data.graphset import Graphset
from data.graph_sampler import AttributeSampler
from data.ced_dataset import CEDDataset
from data.esposalles_dataset import EsposallesDataset
from data.iam_dataset import IAMDataset
import data.volumes as dv

## Model Things
from models import visual_encoders as VE



### Utils
import utils 
import visualizations as visu

## Pipelines
import pipelines as pipes


## Common packages
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

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

        
def batch_step(loader: Type[DataLoader],
               model: Type[nn.Module],
               criterion: Type[nn.Module],
               optimizer: Type[torch.optim.Adam]):
    
    model.train()

    batch_loss = 0

    for idx, batch in tqdm.tqdm(enumerate(loader), desc="Batch Loop", leave=True, position=1):
        optimizer.zero_grad()
        images = batch["image_lines"].to(device)
        original_images = batch["non_augmented_image_lines"].to(device)
        visu.plot(original_images.cpu())
        exit()
        ocr = batch["ocrs"]

        reconstructed_image = model(images)

        loss = criterion(original_images, reconstructed_image)
        batch_loss += loss

        loss.backward()
        optimizer.step()

    epoch_loss = batch_loss/len(loader)
    print(epoch_loss)
    exit()
    wandb.log({"Training Loss": epoch_loss})

    return epoch_loss

@torch.no_grad()
def eval(loader: Type[DataLoader],
               model: Type[nn.Module],
               criterion: Type[nn.Module]):
    model.eval()

    batch_loss = 0

    for idx, batch in tqdm.tqdm(enumerate(loader), desc="Validation/Test Batch Loop", leave=True, position=1):
        images = batch["image_lines"].to(device)
        original_images = batch["non_augmented_image_lines"].to(device)
        ocr = batch["ocrs"]

        reconstructed_image = model(images)

        loss = criterion(original_images, reconstructed_image)
        batch_loss += loss


    epoch_loss = batch_loss/len(loader)

    wandb.log({"Validation Loss": epoch_loss})

    return epoch_loss


@hydra.main(config_path="./configs", config_name="pretrain", version_base="1.3.2")
def main(cfg: DictConfig):
    CFG_DATA = cfg.data
    CFG_MODELS = cfg.models
    CFG_SETUP = cfg.setup

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
    list_of_datasets = []

    for dataset_object in (CFG_DATA.datasets):
        data = instantiate(CFG_DATA.datasets[dataset_object])
        list_of_datasets.append(data)
        means_heigth.append(np.mean(data.line_heights))
        means_width.append(np.mean(data.line_widths))

    standarized_width = ceil(np.mean(means_width))
    standarized_height = ceil(np.mean(means_heigth))

    for dts in list_of_datasets:
        dts.define_transforms((standarized_height, standarized_width))
    
    
    if len(list_of_datasets) == 1:
        print("Only training with one Dataset")
        merged_dataset = list_of_datasets[-1]
    else:
        merged_dataset = torch.utils.data.ConcatDataset(list_of_datasets)
    
    collate_fn = list_of_datasets[0].collate_fn

    generator = torch.Generator().manual_seed(2)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(merged_dataset, partitions_ratio, generator=generator)
    # *

    #visu.plot([list_of_datasets[0][0][0]])
    #exit()
    # * Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=8, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=1, collate_fn=collate_fn)
    # *

    print("DATA LOADER SUCCESFULLY GENERATED")

    model = VE.LineAutoEncoder(cfg=CFG_MODELS.visual_encoder).to(device)

    if CFG_MODELS.finetune is True:
        model_name = f"./checkpoints/{checkpoint_name}.pt"
        
        model.load_state_dict(torch.load(model_name))
        print("Model Loaded Succesfully Starting Finetunning")

    optimizer = hydra.utils.instantiate(CFG_SETUP.optimizer, params=model.parameters())
    criterion = nn.MSELoss()

    optimal_loss = 10000

    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_name = f"./checkpoints/{checkpoint_name}.pt"


    ## Start the training
    print("CREATING THE BASELINE METRIC VALUE\n STARTING TO EVALUATE FO THE FIRST TIME")
    loss_validation = eval(loader=validation_loader,
                           model=model,
                           criterion=criterion)
    
    print(f"Validation Loss Epoch: {0} Value: {loss_validation}")


    _, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                actual_metric=loss_validation, 
                                model=model, 
                                checkpoint_path=checkpoint_name, 
                                compare="<")
    
    #wandb.log({"Validation Loss": loss_validation})
    
    for epoch in tqdm.tqdm(range(epochs), desc="Training Process", position=0, leave=False):
        
    
        train_loss = batch_step(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer
        ) ## to fulfill

        print(f"Loss Epoch: {epoch} Value: {train_loss}")
        if ((epoch +1) % 10) == 0:
            loss_validation = eval(loader=validation_loader,
                                model=model,
                                criterion=criterion)

            updated, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                        actual_metric=loss_validation, 
                                        model=model, 
                                        checkpoint_path=checkpoint_name, 
                                        compare="<")
            
            if updated:
                print(f"Model Updated: Epoch: {epoch}")
            
            #wandb.log({"Validation Loss": loss_validation})

                    
    model.load_state_dict(torch.load(checkpoint_name))

    loss_test = eval(loader=test_loader,
                     model=model,
                     criterion=criterion)
    
    updated, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                actual_metric=loss_test, 
                                model=model, 
                                checkpoint_path=checkpoint_name, 
                                compare="<")
    
    if updated:
        print(f"Model Updated on Test")



if __name__ == "__main__":
    main()