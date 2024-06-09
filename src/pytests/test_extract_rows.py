import pytest
import json

import numpy as np

from src.tasks.extract_rows import *
from PIL import Image





class TestClass:
    
    def pipeline(self, IMAGE_PATH):
        
        image = Image.open(IMAGE_PATH).convert("RGB")
        print("Image with Shape", np.array(image).shape)

        graph_gt_path = os.path.dirname(IMAGE_PATH)
        
        with open(graph_gt_path + "/graph_gt.json", "rb") as file:
            gt = json.load(file)

        num_integrants = gt[os.path.basename(IMAGE_PATH)]["individus"]
        

        ## EXAMPLE OF THE FUNCTIONALITY OF EACH FCNTION

        ## TABLE EXTRACTION
        table_coordinates = extract_table_coordinates(images=[image])
        x, y, w, h = table_coordinates[-1]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        img = np.array(image)
        cropped_image = img[y:y+h, x:x+w]
        cropped_image = Image.fromarray(cropped_image)

        ## ROWS EXTRACTION
        bboxes_coordinates = extract_bboxes_coordinates(images=[cropped_image], return_polygons=False)
        max_width_image = np.array(cropped_image).shape[1]

        ## Final lines extraction
        rows = extract_rows(bboxes=bboxes_coordinates, max_image_width=max_width_image, return_polygon=False)
        
        if len(rows[0]) < num_integrants:
            rows = substract_headers(image=image, filtered_lines=rows[0])
            return rows, num_integrants
        
        return rows, num_integrants
        
    
    def test_1881(self):
        
        IMAGE_PATH = "data/Documents/SFLL/1881/050000120052046,0009.jpg"
        
        rows, num_integrants = self.pipeline(IMAGE_PATH)   
        
        assert len(rows[0]) ==  num_integrants


        
    def test_1889(self):
        IMAGE_PATH = "data/Documents/SFLL/1889/050000120052048,0020.jpg"
        rows, num_integrants = self.pipeline(IMAGE_PATH)   
        assert len(rows[0]) ==  num_integrants
        
    def test_1906(self):
        IMAGE_PATH = "data/Documents/SFLL/1906/050000120052053,0010.jpg"
        rows, num_integrants = self.pipeline(IMAGE_PATH)   
        assert len(rows[0]) ==  num_integrants
        
        


  