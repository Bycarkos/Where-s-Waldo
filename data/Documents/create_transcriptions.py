import copy
import time
import math

import os
import numpy as np
import pandas as pd

from typing import *

import matplotlib.pyplot as plt


DATA = pd.read_excel("data/Documents/Base_5_municipis_30_05_2022.xlsx")

BASE_PATH = "data/Documents/SFLL"
YEARS = sorted(os.listdir(BASE_PATH))
transcripted_years = [str(i) for i in sorted(DATA["any_padro"].unique())]

YEARS = sorted(list(set(YEARS) & set(transcripted_years)))


MUNICIPI = os.path.basename(BASE_PATH)
print(MUNICIPI)
print(YEARS)

municipi_population = DATA.loc[(DATA["Municipi"] == MUNICIPI)] 

#saving population in the municipal census
municipi_population.to_csv(os.path.join(BASE_PATH,"SFLL.csv"), sep=",", index="False")


for year in YEARS:


    population_year = municipi_population.loc[(municipi_population["any_padro"] == int(year))]
    

    year_path = os.path.join(BASE_PATH, year)

    population_year.to_csv(os.path.join(year_path,"gt.csv"), sep=",", index=False)