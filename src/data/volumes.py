## custom imports


## Typing packages
from typing import *
from pathlib import Path

from dataclasses import dataclass, field
import os
import re
import cv2 as cv

from PIL import Image
import pandas as pd 



def sort_key(filename):
    # Extract the row part
    row_part = filename.split('/')[-1].replace('.jpg', '')
    # Extract numbers from the row part
    parts = re.split(r'[_]', row_part)
    # Convert parts to integers where possible
    return [int(part) if part.isdigit() else part for part in parts]



@dataclass
class Line:

    _path: Path
    _bbox: Tuple[int, ...]
    _relative_position: Tuple[int, ...]
    _ocr: Tuple[str, ...]


    def __len__(self):
        return len(self._ocr)
    

    def image(self):
        img = cv.imread(self._path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img    

@dataclass
class Page:

    _path : Path
    _bbox: Tuple[int, ...]
    _individuals: List[Type[Line]]

    _n_families: int



    def __len__(self):
        return len(self._individuals)
    

    def image(self):
        img = cv.imread(self._path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img
    
    def gt(self) -> pd.DataFrame:
        base_folder = self._path.with_suffix("")
        return pd.read_csv(Path(base_folder, "gt_alignement.csv"))

@dataclass
class Volume:
    _path: Path
    _pages: List[Type[Page]]
    _year: str
    _entities: Tuple[str] = field(default_factory=lambda: ("nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"))


    def count_attributes(self):
        total_attributes = 0

        for page in self._pages:
            total_attributes += len(self._entities) * len(page)

        return total_attributes
    
    def count_individuals(self):
        total_individuals = 0
        for page in self._pages:
            total_individuals += len(page)

        return total_individuals



    def gt(self):
        return pd.read_csv(Path(self._path, "gt.csv"))
    
    def __len__(self):
        return len(self._pages)
    

