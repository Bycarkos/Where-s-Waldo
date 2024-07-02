## custom imports


## Typing packages
from typing import *
from pathlib import Path

from dataclasses import dataclass, field
import os
import re

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
        return Image.open(self._path).convert("RGB")
    

@dataclass
class Page:

    _path : Path
    _bbox: Tuple[int, ...]
    _individuals: List[Type[Line]]

    _n_families: int



    def __len__(self):
        return len(self._individuals)
    

    def image(self):
        return Image.open(self._path).convert("RGB")
    
    def gt(self) -> pd.DataFrame:
        base_folder = self._path.with_suffix("")
        return pd.read_csv(Path(base_folder, "gt_alignement.csv"))

@dataclass
class Volume:
    _path: Path
    _pages: List[Type[Page]]
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
    


    

    
if __name__ == "__main__":
    import json
    import glob

    volume_1889 = Path("data/CED/SFLL/1889")

    entities = ["nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"]

    ## volumes structure
    volums =  []

    ## Pages Structure
    
    pages_path = sorted([f.path for f in os.scandir(volume_1889) if f.is_dir()])
    with open(Path(volume_1889, "graph_gt_corroborator.json")) as file:
        graph_gt = json.load(file)

    count= 0
    pages = []
    for idx, page_folder in enumerate(pages_path):
        print(page_folder)
        gt_page = pd.read_csv(page_folder + "/gt_alignement.csv")


        ##extract the information of the lines first
        # condition to remove pages with inconsitencis
        page_lines = []
        page = Image.open(page_folder+".jpg")

        n_families = graph_gt[os.path.basename(page_folder)+".jpg"]["families"]

        if len(glob.glob(os.path.join(page_folder, "row_*"))) != graph_gt[os.path.basename(page_folder)+".jpg"]["individus"]:
            count += 1
            continue
        
        else:
            lines_page = (glob.glob(os.path.join(page_folder, "row_*")))
            sorted_lines = sorted_file_names = sorted(lines_page, key=sort_key)
            with open(page_folder + "/info.json", "rb") as file:
                bboxes = json.load(file)["rows_bbox"]

            for idx_line, path_line in enumerate(sorted_lines):
                ocr = gt_page.loc[idx_line, entities].values
                bbox_line = bboxes[idx_line]
                page_lines.append(Line(Path(path_line), bbox_line, bbox_line, ocr))



        pages.append(Page(Path(page_folder+".jpg"), [0, 0, *page.size ], page_lines, n_families))
        
    volums.append(Volume(volume_1889, pages, entities))
