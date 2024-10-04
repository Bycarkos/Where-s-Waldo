## Dataset Things
from data.volumes import Volume, Page, Line
from data.graphset import Graphset
from data.graph_sampler import GraphSampler, AttributeSampler
from data.ced_dataset import CEDDataset

import data.volumes as dv

## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as VE
from models import edge_visual_encoders as EVE
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


## Typing Packages
from typing import *
from pytorch_metric_learning import miners, losses


## Configuration Package
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

## Experiment Tracking packages
import wandb
import tqdm


## Common packages
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from networkx.algorithms import bipartite


import umap
import umap.plot

device = "cuda" if torch.cuda.is_available() else "cpu"
from beeprint import pp as bprint
import pdb




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


"""
                if mode == "validation" or mode == "test":
                    previous_x_dict = graph_structure.x_dict   
                    updated_dict = x_dict 
                    
                    dirname = f"./plots/shifts/{attribute}/{mode}"
                    name_file = f"{attribute}_epoch_{epoch}_{idx}_drift.jpg"
                    fig_path = os.path.join(dirname, name_file)
                    dict_drift[name_file] = population.cpu().numpy()
                    
                    
                    
                    x_previous = previous_x_dict[attribute].detach().cpu().numpy()
                    x_after = updated_dict[attribute].detach().cpu().numpy()
                            
                    evaluate_shift(x_previous=x_previous, x_after=x_after, population=population.cpu().numpy(), fig_name=fig_path)
      
      
            if mode == "validation" or mode == "test":
                with open(f"{pickles_path}/drift_corresp.pkl", "ab") as file:
                    pickle.dump(dict_drift, file)          
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
            notes = CFG_SETUP.wandb.notes,
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
    number_volum_years = len(CFG_DATA.dataset.volumes) 
    checkpoint_name = CFG_MODELS.name_checkpoint
    optimize = CFG_SETUP.configuration.optimize_task
    embedding_size = CFG_MODELS.common_configuration.embedding_size
    # ^ 

    ## & Extract the dataset and the information in this case 
    volumes = pipes.load_volumes(cfg=CFG_DATA) 
    image_dataset = CEDDataset(Volumes=volumes, cfg=CFG_DATA.dataset)

    df_transcriptions = image_dataset._total_gt
    n_different_individuals = image_dataset._total_individual_nodes
    graphset = Graphset(total_nodes=n_different_individuals,
                        df_transcriptions=df_transcriptions,
                        n_volumes=len(volumes),
                        graph_configuration=CFG_DATA.graph_configuration,
                        auxiliar_entities_pk = pk)

    sampler = AttributeSampler(graph=graphset._graph, batch_size=batch_size, shuffle=shuffle)

    print("Generating DataLoader")

    train_loader = DataLoader(dataset=image_dataset, 
                            batch_size = batch_size,
                            sampler=sampler._train_population,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            pin_memory=True)

    validation_loader = DataLoader(dataset=image_dataset, 
                            batch_size = batch_size,
                            sampler=sampler._validation_population,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            pin_memory=True)

    test_loader = DataLoader(dataset=image_dataset, 
                            batch_size = batch_size,
                            sampler=sampler._test_population,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            pin_memory=True)
                            

    total_loader = DataLoader(dataset=image_dataset, 
                            batch_size = batch_size,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            shuffle=True,
                            pin_memory=True)

    
    #? 
    print("DATA LOADED SUCCESFULLY")
    ## ^ Model 

    H = image_dataset.general_height 
    W = image_dataset.general_width // 16

    CFG_MODELS.edge_visual_encoder.input_atention_mechanism = H*W

    model = MMGCM(visual_encoder=VE.LineFeatureExtractor, gnn_encoder=gnn.AttributeGNN, edge_encoder=EVE.EdgeAttFeatureExtractor, cfg=CFG_MODELS).to(device)
    
    if CFG_MODELS.finetune is True:
        model_name = f"./checkpoints/{checkpoint_name}.pt"
        
        model.load_state_dict(torch.load(model_name))
        print("Model Loaded Succesfully Starting Finetunning")
    
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

    if CFG_MODELS.add_language is True:
        
        filepath  = f"embeddings/language_embeddings_{number_volum_years}_{embedding_size}_entities_{len(CFG_SETUP.configuration.compute_loss_on)}.pkl"
        
        if os.path.exists(filepath):
            language_embeddings = utils.read_pickle(filepath)
                
        else:
            language_embeddings = utils.extract_language_embeddings(list(image_dataset._map_ocr.keys()), embedding_size=embedding_size, filepath=filepath)
    
        print("LANGUAGE EMBEDDINGS EXTRACTED SUCCESFULLY")
    
        core_graph.x_language = language_embeddings

        print(image_dataset._map_ocr)


    core_graph.epoch_populations = image_dataset._population_per_volume
    
    
    
    criterion = losses.TripletMarginLoss(margin=0.2,
                    swap=False,
                    smooth_loss=False,
                    triplets_per_anchor="all")

                    
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        if epoch == 0:
            print("SAVING TRAINING MODEL")
            os.makedirs(f"./checkpoints/{checkpoint_name}", exist_ok=True)
            model_name = f"./checkpoints/{checkpoint_name}/{checkpoint_name}_train_{epoch}.pt"
            torch.save(model.state_dict(), model_name)
            
            print("EVALUATING STEP")
            validation_loss = pipes.evaluate(
                                    loader=validation_loader,
                                    graph=core_graph, 
                                    model=model, 
                                    criterion=criterion,
                                    entities=nodes_to_compute_loss,
                                    language_distillation=CFG_MODELS.add_language)

            
            wandb.log({"Validation Loss": validation_loss})
            if validation_loss < optimal_loss:
                print("UPDATING THE MODEL...")
                os.makedirs("./checkpoints", exist_ok=True)
                model_name = f"./checkpoints/{checkpoint_name}.pt"


                torch.save(model.state_dict(), model_name)
                optimal_loss = validation_loss

        loss  = pipes.batch_step(loader=train_loader,
                                graph=core_graph, 
                                model=model, 
                                criterion=criterion, 
                                optimizer=optimizer,
                                entities=nodes_to_compute_loss, 
                                epoch=epoch)

        
        if (epoch +1) % 10 == 0:
            print("SAVING TRAINING MODEL")
            os.makedirs(f"./checkpoints/{checkpoint_name}", exist_ok=True)
            model_name = f"./checkpoints/{checkpoint_name}/{checkpoint_name}_train_{epoch}.pt"
            torch.save(model.state_dict(), model_name)

            validation_loss = pipes.evaluate(
                                    loader=validation_loader,
                                    graph=core_graph, 
                                    model=model, 
                                    criterion=criterion,
                                    entities=nodes_to_compute_loss,
                                    language_distillation=CFG_MODELS.add_language)            
            
            wandb.log({"Validation Loss": validation_loss})

            if validation_loss < optimal_loss:
                print("UPDATING THE MODEL...")

                os.makedirs("./checkpoints", exist_ok=True)
                model_name = f"./checkpoints/{checkpoint_name}.pt"

                torch.save(model.state_dict(), model_name)
                optimal_loss = validation_loss

             


    model.load_state_dict(torch.load(model_name))
    
    test_loss = pipes.evaluate(
                            loader=test_loader,
                                    graph=core_graph, 
                                    model=model, 
                                    criterion=criterion,
                                    entities=nodes_to_compute_loss,
                                    language_distillation=CFG_MODELS.add_language)
                
    if test_loss < optimal_loss:
        print("UPDATING THE MODEL...")
        os.makedirs("./checkpoints", exist_ok=True)
        model_name = f"./checkpoints/{checkpoint_name}.pt"

        torch.save(model.state_dict(), model_name)
    
    wandb.finish()

        
        
if __name__ == "__main__":

    main()
        