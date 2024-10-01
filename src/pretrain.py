## Dataset Things
from data.volumes import Volume, Page, Line
from data.graphset import Graphset
from data.graph_sampler import AttributeSampler
from data.image_dataset import ImageDataset
from data.esposalles_dataset import EsposallesDataset


import data.volumes as dv

## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as VE
from models import edge_visual_encoders as EVE
from models.graph_construction_model import MMGCM



### Utils
import utils 


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

from beeprint import pp as bprint
import pdb

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf


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

        

        
if __name__ == "__main__":


    overrides = [
    ]


    with initialize(version_base="1.3.2", config_path="./configs"):
        CFG = compose(config_name="pretrain", overrides=overrides, return_hydra_config=True)
    

    data = EsposallesDataset(path="/home/cboned/data/HTR/Esposalles")

    model = VE.LineAutoEncoder(CFG.models.visual_encoder).to(device)   

    codification, to_decode = model.encoder(data[0][0].unsqueeze(0).to(device))
    model.decoder(to_decode)