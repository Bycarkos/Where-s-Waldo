import cv2 as cv
import numpy as np

from surya.detection import batch_text_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.settings import settings

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple
from scipy import stats

try:
    import src.tasks.task_utils as utils
except:
    import task_utils as utils

import os
import glob


RESOLUTION_IMAGE = (96, 224)
QUERIES = ['A piece of an old handwritten document with pen or pencil .',
            'A piece of an old typewritted document with typewritting machine.',# document with typewritting machine.',
            'An empty  document with.']

DISTANCE_PROPORTIONS = {
    1924: 40, 
    1906: 10,
    1889: 15,
    1930: 20,
    1910: 15,
    1915: 15
}


CLIP_PROCESS = {
    1906: (64, 128),
    1889: (64, 128),
    1915: (96, 224)
}


def substract_headers(image: Image.Image, filtered_lines: List[Tuple[int, ...]], resolution_image:Tuple[int, int]) -> List[Tuple[int, ...]]:
    
    
    M, N = resolution_image

    if isinstance(image, Image.Image):
        image = np.array(image)

    masked_image = utils.segment_image(image,
                                       M=M,
                                       N=N,
                                       queries=QUERIES)
    

    flines = sorted(filtered_lines, key=lambda x: x[1])

    final_filtered_lines = []
    for line_bbox in flines:
        x,y,w,h = line_bbox
        typewritting_sum = np.sum(masked_image[y:y+h,x:x+w,1])
        handwritten_sum = np.sum(masked_image[y:y+h,x:x+w,0])
        blank_sum = np.sum(masked_image[y:y+h,x:x+w,2])
        
        if (handwritten_sum > typewritting_sum + (typewritting_sum*0.2)) and ((handwritten_sum > blank_sum + (blank_sum*0.2))):
            final_filtered_lines.append(line_bbox)

    return [final_filtered_lines]


def extract_table_coordinates(images: List[Image.Image]) -> List[Image.Image]:

    """
    Extracts coordinates of tables from a list of images.

    This function processes a list of images to detect tables. It uses a pre-trained 
    model for layout detection and another model for text detection. The coordinates 
    of detected tables are returned as a list.

    Args:
        images (List[Image.Image]): List of input images for table detection.

    Returns:
        List[np.ndarray]: List of bounding box coordinates for detected tables in the images.
    
    Raises:
        AssertionError: If the input `images` is not a list.
    """

    assert isinstance(images, list), "the input images: must be a list of images"

    table_coordinates = []

    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    line_predictions = batch_text_detection(images, det_model, det_processor)
    layout_predictions = batch_layout_detection(images, model, processor, line_predictions)


    for layout_bbox in layout_predictions:
        reference_area = layout_bbox.image_bbox[2] * layout_bbox.image_bbox[3] 
        temporal_bbox = []
        for bbox_object in layout_bbox.bboxes:
            if bbox_object.label == "Table":
                area_table = bbox_object.bbox[2] * bbox_object.bbox[3] 
                if reference_area < 2*area_table:
                    temporal_bbox.append(bbox_object.bbox)
                
                
        if (len(temporal_bbox) > 1) or (len(temporal_bbox) == 0):
            
            table_coordinates.append(layout_bbox.image_bbox)
            
        else:
            table_coordinates.append(temporal_bbox[0])
        

    return table_coordinates





def extract_bboxes_coordinates(images: List[Image.Image], return_polygons:bool=True) -> List[np.ndarray]:

    """
    Extracts bounding box coordinates and optionally their corresponding polygons from a list of images.

    This function takes a list of images, performs text detection using a pre-trained model, and 
    returns the bounding boxes for detected text regions. Optionally, it can also return the 
    polygon coordinates for these regions.

    Args:
        images (List[Image.Image]): List of input images for text detection.
        return_polygons (bool): Flag to indicate whether to return polygons along with bounding boxes. 
                                Defaults to True.

    Returns:
        List[np.ndarray]: If return_polygons is True, returns a tuple of two lists: 
                          - List of bounding boxes for each image.
                          - List of polygon coordinates for each bounding box.
                          If return_polygons is False, returns only the list of bounding boxes.
    
    Raises:
        AssertionError: If the input `images` is not a list.
    """


    assert isinstance(images, list), "the input images: must be a list of images"


    image_polygons = []
    image_bboxes = []

    model, processor = load_model(), load_processor()

    predictions = batch_text_detection(images, model, processor)

    for lay_bbox in predictions:
        polygons_per_lay_bbox = [i.polygon for i in lay_bbox.bboxes]
        bboxes_per_lay_bbox = [utils.extract_bounding_box(i) for i in polygons_per_lay_bbox]
        image_polygons.append(polygons_per_lay_bbox)
        image_bboxes.append(bboxes_per_lay_bbox)

    if return_polygons:
        return image_bboxes, image_polygons
    
    else:
        return image_bboxes



def extract_rows(bboxes:List[List[np.ndarray]], max_image_width:int, confidence_interval,  return_polygon:bool=True):


    """
    Extracts and orders rows of bounding boxes from a list of images.

    This function processes a list of bounding boxes associated with images to extract 
    rows of text or objects. It does so by first sorting the bounding boxes based on 
    their y-coordinates, then filtering, merging horizontally aligned boxes, and 
    finally ordering the merged boxes to form lines.

    Args:
        images (List[Image.Image]): List of images corresponding to the bounding boxes.
        bboxes (List[List[np.ndarray]]): List of bounding boxes for each image, where 
                                         each bounding box is represented as a numpy array.

    Returns:
        List[np.ndarray]: A list of merged and ordered bounding boxes representing rows.
    """

    image_lines = []
    image_lines_polygons = []
    for (image_bboxes) in (bboxes):
            
        x_variation = [i[0] for i in sorted(image_bboxes, key=lambda x: x[0])]

        extract_xaxis_variance = [i for i in sorted(image_bboxes, key=lambda x: x[0]) if i[0] <= np.std(x_variation)*1.2]
        

        merged_boxes = utils.merge_horizontally_aligned_bounding_boxes(sorted(extract_xaxis_variance, key=lambda x: x[1]), confidence_interval=20)
        merged_boxes = utils.filter_bounding_boxes(merged_boxes)
        count_bboxes_per_example = []

        bboxes_filtered = np.array(merged_boxes)
        max_horizontal_scanning = max_image_width//2

        for column in (range(0, max_horizontal_scanning, 10)):
            bool1 = (bboxes_filtered[:, 0] < column)
            bool2 = ((bboxes_filtered[:, 0] +   bboxes_filtered[:, 2]) > column )
            bboxes_in_that_column = np.logical_and(bool1, bool2)
            count_bboxes_per_example.append(bboxes_filtered[bboxes_in_that_column])

        final_bboxes = sorted(count_bboxes_per_example, key = lambda x: len(x), reverse=True)[0]

        # Extract the next coordinates that are far away from tyhe content (headers)
        ordered_merged_boxes = sorted(final_bboxes, key=lambda x: x[1])

        for _ in range(2):
            lines = utils.get_lines_from_bboxes(ordered_merged_boxes, image_max_width=max_image_width, confidence_interval=confidence_interval)
            ordered_merged_boxes = sorted(lines, key=lambda x: x[1])
            
        image_lines.append(ordered_merged_boxes)    

        if return_polygon:
            lines_pol = [utils.generate_polygon(*i) for i in lines]
            image_lines_polygons.append(lines_pol)
    
    if return_polygon:
        return image_lines, image_lines_polygons
    else:
        return image_lines





def main(list_years:List, basepath:str):

    possible_error_documents = []
    for idx, year in enumerate(list_years):

        preprocess_clip = True if CLIP_PROCESS.get(year, None) is not None else False 

        year = str(year)
        print("Extracting information from year: ", year )
        abs_path = os.path.join(basepath,year)
        print("Total of pages: ", len(glob.glob(abs_path + "/*.jpg")))


        with open(abs_path + "/graph_gt.json", "r") as file:
            gt = json.load(file)

        for document, info in gt.items():
            custom_information = {

            }
            num_integrants = info["individus"]
            name_document = os.path.splitext(document)[0]
            
            auxiliar_path = os.path.join(abs_path, name_document) 

            if os.path.exists(auxiliar_path):
                continue
            
            os.makedirs(auxiliar_path, exist_ok=True)


            IMAGE_PATH = os.path.join(abs_path, document)

            image = Image.open(IMAGE_PATH).convert("RGB")

            h, w, _ = np.array(image).shape

            if h+w < 5000:
                table_coordinates = extract_table_coordinates(images=[image])
                x, y, w, h = table_coordinates[-1]

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                cropped_image = np.array(image)[y:y+h, x:x+w]

            else:
                cropped_image = np.array(image)


            
            cropped_image = Image.fromarray(cropped_image)
            bboxes_coordinates= extract_bboxes_coordinates(images=[cropped_image], return_polygons=False)

            if preprocess_clip:
                bboxes_coordinates = substract_headers(image=cropped_image, filtered_lines=bboxes_coordinates[0], resolution_image=CLIP_PROCESS[int(year)])
            
            max_width_image = np.array(cropped_image).shape[1]
            confidence_interval = DISTANCE_PROPORTIONS[int(year)]
            images_rows, pol = extract_rows(bboxes=bboxes_coordinates, max_image_width=max_width_image, confidence_interval=confidence_interval, return_polygon=True)


            lines_out = utils.draw_polygons(np.array(cropped_image)/255, polygons=pol)
            image_uint8 = (lines_out * 255).astype(np.uint8)

            for rows in images_rows:
                custom_information["rows_bbox"] = list(rows)

                
                if len(rows) != num_integrants:
                    if len(rows) > num_integrants:
                        print(f"POSSIBLE HEADER ERROR IN {document}")
                        possible_error_documents.append(document+ '_possible_header_error')
                    else:
                        print(f"POSSIBLE MISSING ERROR IN {document}")
                        possible_error_documents.append(document+ "_possible_missing_error")
                        

                    custom_information["possible_error"] = True
                else:
                    custom_information["possible_error"] = False



                ## Save the information
                with open(os.path.join(auxiliar_path, "info.json"), "a") as file:
                    json.dump(custom_information, file, default=int)


                    
                cropped_image = np.array(cropped_image)
                for idx, row in enumerate(rows):
                    x, y, w, h = row

                    cropped_line = cropped_image[y:y+h, x:x+w]

                    name = f"row_{idx}.jpg"

                    cv.imwrite(filename=os.path.join(auxiliar_path, name), img=cropped_line[:,:,::-1])

                cv.imwrite(filename=os.path.join(auxiliar_path, "output_image.jpg"), img=image_uint8[:,:,::-1])
                
        with open(os.path.join(abs_path, "Error.txt"), "w") as file:
            for line in possible_error_documents:
                file.write(f"{line}\n")



if __name__ == "__main__":
    import json

    main(list_years=[1915], basepath="data/Documents/SFLL")
    exit()





    IMAGE_PATH = "data/Documents/SFLL/1881/050000120052046,0009.jpg"

    image = Image.open(IMAGE_PATH).convert("RGB")
    print("Image with Shape", np.array(image).shape)

    with open("data/Documents/SFLL/1924/graph_gt.json", "rb") as file:
        gt = json.load(file)

    num_integrants = gt["050000120052059,0021.jpg"]["individus"]
    

    ## EXAMPLE OF THE FUNCTIONALITY OF EACH FCNTION

    ## TABLE EXTRACTION
    table_coordinates = extract_table_coordinates(images=[image])
    print(table_coordinates)
    x, y, w, h = table_coordinates[-1]
    img = np.array(image)
    print()
    cropped_image = img[y:y+h, x:x+w]
    cropped_image = Image.fromarray(cropped_image)
    plt.imshow(cropped_image)
    plt.show()

    ## ROWS EXTRACTION
    bboxes_coordinates, polygon_coordinates = extract_bboxes_coordinates(images=[cropped_image], return_polygons=True)
    max_width_image = np.array(cropped_image).shape[1]
    image_out = utils.draw_polygons(image=np.array(cropped_image)/255, polygons=polygon_coordinates[0])
    image_uint8 = (image_out * 255).astype(np.uint8)
    imagee = Image.fromarray(image_uint8)
    plt.imshow(imagee)
    plt.show()

    ## Final lines extraction
    rows, pol = extract_rows(bboxes=bboxes_coordinates, max_image_width=max_width_image)
    print(len(rows[0]))
    len(num_integrants)
    if len(rows[0]) < num_integrants:
        print(len(rows[0]))
        print(num_integrants)
        rows = substract_headers(image=image, filtered_lines=rows[0])
    lines_out = utils.draw_polygons(np.array(cropped_image)/255, polygons=pol)
    image_lines_uint8 = (lines_out * 255).astype(np.uint8)
    plt.imshow(image_lines_uint8)
    plt.show()
    exit()
