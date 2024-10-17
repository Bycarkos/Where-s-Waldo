## Dataset Things
import torch.utils
import torch.utils.data
from data.graphset import Graphset
from data import CEDDataset, WashingtonDataset, IAMDataset, EsposallesDataset


## Model Things
from models.gnn_encoders import AttributeGNN
from models.visual_encoders import LineAutoEncoder
from models.edge_visual_encoders import DisentanglementAttentionEncoder
from models.graph_construction_model import MMGCM


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
from pytorch_metric_learning import miners, losses
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


## Typing Packages
from typing import *
from jaxtyping import Float, Array

## Configuration Package
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

## Experiment Tracking packages
import wandb
import tqdm


## Common packages
import os
from beeprint import pp as bprint
import pdb
from math import ceil
import numpy as np


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

    
@hydra.main(config_path="./configs", config_name="train", version_base="1.3.2")
def main(cfg: DictConfig):
    
    CFG_MODELS = cfg.models
    CFG_DATA = cfg.data
    CFG_SETUP    = cfg.setup

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

        #wandb.init(**dict(CFG_SETUP.wandb))

    
    #! The utility of this dictionary is to relate the groundtruth with the visual information AD-HOC
    pk = {"Noms_harmo":"nom", "cognom1_harmo":"cognom_1", "cognom2_harmo":"cognom_2", "parentesc_har":"parentesc", "ocupacio":"ocupacio"}
    
    #  ^ Hydra things
    epochs = CFG_SETUP.configuration.epochs
    batch_size = CFG_DATA.collator.batch_size
    shuffle = CFG_DATA.collator.shuffle
    partitions_ratio = CFG_DATA.collator.partitions_ratio
    image_embedding_size = CFG_MODELS.visual_encoder.model.output_channels
    # ^ 

    ## & Extract the dataset and the information in this case 
    ### Ced dataset needs path, volume_years, attributes
    path_dataset = CFG_DATA.dataset.path
    volumes_years = CFG_DATA.dataset.volumes_years
    attributes = CFG_DATA.dataset.attributes
    image_dataset = CEDDataset(path=path_dataset, volumes_years=volumes_years, attributes=attributes)
    
    standarized_width = ceil(np.mean(image_dataset.line_widths))
    standarized_height = ceil(np.mean(image_dataset.line_heights))
    

    
    image_dataset.define_transforms(new_shape=(standarized_height + 100, standarized_width))
    
    ## & Extract the dataset and the information in this case 
    
    
    ## & Generator
    generator = torch.Generator().manual_seed(2)
    ## & Generator
    
    ## & Dataset random partitions
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(image_dataset, partitions_ratio, generator=generator)
    ## & Dataset random partitions
    
    
    ## & Graph Dataset
    df_transcriptions = image_dataset._total_gt
    n_different_individuals = image_dataset._total_individual_nodes
    graphset = Graphset(total_nodes=n_different_individuals,
                        df_transcriptions=df_transcriptions,
                        n_volumes=len(volumes_years),
                        graph_configuration=CFG_DATA.graph_configuration,
                        auxiliar_entities_pk = pk)
    
    
    ## & Graph Dataset
    

    print("Generating DataLoader")

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size = batch_size,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=8,
                            pin_memory=True,
                            shuffle=shuffle)

    validation_loader = DataLoader(dataset=validation_dataset, 
                            batch_size = 1,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size = 1,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            pin_memory=True)
                            

    total_loader = DataLoader(dataset=image_dataset, 
                            batch_size = 1,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            shuffle=False,
                            pin_memory=True)

    
 
    print("DATA LOADED SUCCESFULLY")
    ## ^ Model 


    ## ^ Vision Extractor Model
    
    VISUAL_ENCODER = instantiate(CFG_MODELS.visual_encoder.model)
    
    if CFG_MODELS.visual_encoder.finetune == True:
        checkpoint_name = CFG_MODELS.visual_encoder.checkpoint
        VISUAL_ENCODER.load_state_dict(torch.load(checkpoint_name))
        print("Visual model Loaded Succesfully Starting Finetunning")
        
        if CFG_MODELS.visual_encoder.freeze:
            for params in VISUAL_ENCODER.parameters():
                params.requires_grad = False

        
    ## ^ Vision Extractor Model
    
    ## ^ Edge Disentanglement model

    EDGE_MODEL = instantiate(CFG_MODELS.edge_visual_encoder.model)

    if CFG_MODELS.edge_visual_encoder.finetune == True:
        checkpoint_name = CFG_MODELS.edge_visual_encoder.checkpoint
        EDGE_MODEL.load_state_dict(torch.load(checkpoint_name))
        print("Edge model Loaded Succesfully Starting Finetunning")
        
    ## ^ Edge Disentanglement model


    ## ^ GNN Model
    
    GNN_MODEL = instantiate(CFG_MODELS.gnn_encoder.model, n_different_edges=len(attributes))
    
    if CFG_MODELS.gnn_encoder.finetune == True:
        checkpoint_name = CFG_MODELS.gnn_encoder.checkpoint
        GNN_MODEL.load_state_dict(torch.load(checkpoint_name))
        print("GNN model Loaded Succesfully Starting Finetunning")
            
    ## ^ GNN Model
    
    
    ## ** Merged Model
    model = MMGCM(visual_encoder=VISUAL_ENCODER,
                  gnn_encoder=GNN_MODEL,
                  edge_encoder=EDGE_MODEL,
                  n_different_edges=len(attributes)).to(device)
    ## ** Merged Model
    
    print("MODELS LOADED SUCCESFULLY")
    
    optimizer = hydra.utils.instantiate(CFG_SETUP.optimizer, params=model.parameters())

    if cfg.verbose == True:
        print("Configuration of the Models: ")
        bprint(dict(CFG_MODELS))

    if cfg.verbose == True:
        print("Inizialization Done without problemas")
        print("Starting the training proces with the following configuration: ")
        bprint(dict(CFG_SETUP.configuration)) 
    
    ## ^  

    optimal_loss = 10000
    
    nodes_to_compute_loss = CFG_SETUP.configuration.compute_loss_on
    core_graph = graphset._graph


    core_graph.epoch_populations = image_dataset._population_per_volume
    core_graph.x_image_entity = torch.zeros(graphset._total_individual_nodes, image_embedding_size)
    
    
    
    criterion = losses.TripletMarginLoss(margin=0.2,
                    swap=False,
                    smooth_loss=False,
                    triplets_per_anchor="all")
    
    
    ## **Scheduler
    num_warmup_steps = 1300  
    train_loader_len = len(train_loader)  # Assuming you have defined train_loader
    total_steps = epochs * train_loader_len
    num_cycles = epochs // 2
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
        #num_cycles=num_cycles,
    )
    ## **

    optimal_loss = 10000

    checkpoint_name = CFG_MODELS.name_checkpoint

    ## Start the training
    print("CREATING THE BASELINE METRIC VALUE\n STARTING TO EVALUATE FO THE FIRST TIME")

    
    loss_validation = pipes.evaluate(loader=validation_loader,
                                 graph=core_graph,
                                 model=model,
                                 criterion=criterion,
                                 entities=nodes_to_compute_loss,
                                 epoch=0)
    
    _, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                actual_metric=loss_validation, 
                                model=model, 
                                checkpoint_path=checkpoint_name, 
                                compare="<")

    print(f"Validation Loss Epoch: {0} Value: {loss_validation} Optimal_loss: {optimal_loss}")
       
    for epoch in tqdm.tqdm(range(epochs), desc="Training Process", position=0, leave=False):


        train_loss  = pipes.batch_step(loader=train_loader,
                                graph=core_graph, 
                                model=model, 
                                criterion=criterion, 
                                optimizer=optimizer,
                                entities=nodes_to_compute_loss,
                                scheduler=scheduler, 
                                epoch=epoch)

        current_lr = scheduler.get_last_lr()[0]  # Get current learning rate (from scheduler)
        print(f"Loss Epoch: {epoch} Value: {train_loss} LR: {current_lr:.6f}")

        if (epoch +1) % 10 == 0:

            loss_validation = pipes.evaluate(
                                    loader=validation_loader,
                                    graph=core_graph, 
                                    model=model, 
                                    criterion=criterion,
                                    entities=nodes_to_compute_loss,
                                    language_distillation=CFG_MODELS.add_language)            

            updated, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                        actual_metric=loss_validation, 
                                        model=model, 
                                        checkpoint_path=checkpoint_name, 
                                        compare="<")
            if updated:
                print(f"Model Updated: Validation Loss Epoch: {0} Value: {loss_validation} Optimal_loss: {optimal_loss}")
                
                
    
    loss_test = pipes.evaluate(
                loader=test_loader,
                graph=core_graph, 
                model=model, 
                criterion=criterion,
                entities=nodes_to_compute_loss,
                language_distillation=CFG_MODELS.add_language)


    updated, optimal_loss = utils.update_and_save_model(previous_metric=optimal_loss, 
                                actual_metric=loss_test, 
                                model=model, 
                                checkpoint_path=checkpoint_name, 
                                compare="<")
    
    if updated:
        print(f"Model Updated on Test")
    
    wandb.finish()

        
        
if __name__ == "__main__":

    main()
        