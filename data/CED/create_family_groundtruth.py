import copy
import time
import math

import os
import glob
import numpy as np
import pandas as pd

from typing import *

import matplotlib.pyplot as plt

import json



BASE_PATH = "data/CED/SFLL"
YEARS = [1889]#, 1906, 1910, 1915, 1924, 1930]


MUNICIPI = os.path.basename(BASE_PATH)
print(MUNICIPI)
print(YEARS)


"data/CED/SFLL/1889/graph_gt.json"
for year in YEARS:

    with open(os.path.join(BASE_PATH, str(year), "graph_gt_corroborator.json"), "r") as file:
        year_gt = json.load(file)

    actual_year = os.path.join(BASE_PATH, str(year))
    data = pd.read_csv(os.path.join(actual_year, "gt.csv"))
    number_of_families = data.groupby('id_document')['id_padro_llar'].nunique().reset_index()["id_padro_llar"].values
    imatges = sorted(glob.glob(actual_year + "/*.jpg"))
    imatges = [os.path.basename(x) for x in  imatges]
    carrier = 0
    for ind_per_imatge, imatge in zip(number_of_families, imatges):
        name_image = os.path.splitext(imatge)[0]
        path_alignement = os.path.join(actual_year, name_image)
        print(path_alignement)
        year_gt[imatge]["families"] = int(ind_per_imatge)
        
        number_individuals = year_gt[imatge]["individus"]
        print(imatge, number_individuals)
        data_page = data.iloc[carrier: carrier + number_individuals]
        data_page.to_csv(path_alignement + "/gt_alignement.csv")
        carrier += number_individuals 
        print(carrier)

    with open(actual_year + "/graph_gt_corroborator.json", "w") as file:
        json.dump(year_gt, file)

