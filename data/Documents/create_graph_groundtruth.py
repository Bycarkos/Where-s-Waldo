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



BASE_PATH = "data/Documents/SFLL"
YEARS = sorted(os.listdir(BASE_PATH))


MUNICIPI = os.path.basename(BASE_PATH)
print(MUNICIPI)
print(YEARS)

for year in YEARS:

    year_gt = {}
    actual_year = os.path.join(BASE_PATH, year)
    try:
        data = pd.read_csv(os.path.join(actual_year, "gt.csv"))
        individus_per_imatge = data.groupby("id_document").count()["id_padro_individu"].values
        imatges = sorted(glob.glob(actual_year + "/*.jpg"))
        if len(imatges) == 0:
            extra_volumes = os.listdir(actual_year)

            for extra_vol in extra_volumes:
                year_gt = {}

                v = os.path.join(actual_year, extra_vol)
                imatges = sorted(glob.glob(v + "/*.jpg"))
                imatges = [os.path.basename(x) for x in  imatges]

                for ind_per_imatge, imatge in zip(individus_per_imatge, imatges):
                    year_gt[imatge] = {}
                    year_gt[imatge]["individus"] = int(ind_per_imatge)

                with open(v + "graph_gt.json", "a") as file:
                    json.dump(year_gt, file)



        else:
            imatges = [os.path.basename(x) for x in  imatges]

            for ind_per_imatge, imatge in zip(individus_per_imatge, imatges):
                year_gt[imatge] = {}
                year_gt[imatge]["individus"] = int(ind_per_imatge)

            with open(actual_year + "/graph_gt.json", "w") as file:
                json.dump(year_gt, file)

    except:
        continue



    




    #print(data.head())
   # data/Documents/SFLL/1936/UC_156_Seccio_1/13-0662_ES-B_-Ajuntament-de-Sant-Feliu-de-Llobregat-Padró-Gener-1936_00002.jpg
   # data/Documents/SFLL/1936/UC_157_Seccio_2/13-0662_ES-B_-Ajuntament-de-Sant-Feliu-de-Llobregat-Padró-Gen-1936~1_00001.jpg
   # data/Documents/SFLL/1936/UC_158_Seccio_3/13-0662_ES-B_-Ajuntament-de-Sant-Feliu-de-Llobregat-Padró-Gen-1936~2_00001.jpg
   # data/Documents/SFLL/1936/UC_159_Seccio_4/13-0662_ES-B_-Ajuntament-de-Sant-Feliu-de-Llobregat-Padró-Gen-1936~3_00001.jpg
   # data/Documents/SFLL/1936/UC_159_Seccio_4/13-0662_ES-B_-Ajuntament-de-Sant-Feliu-de-Llobregat-Padró-Gen-1936~3_00180.jpg
   # data/Documents/SFLL/1936/UC_161_Seccio_6/13-0662_ES-B_-Ajuntament-de-Sant-Feliu-de-Llobregat-Padró-Gen-1936~5_00001.jpg


   #data/Documents/SFLL/1857_1/050000120052040,0001.jpg
    #data/Documents/SFLL/1857/050000120052041,0001.jpg