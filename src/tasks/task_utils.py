
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

from typing import *

import torch
import clip
import tqdm




def create_visual_label(patch, output_argmax):
    
    zeros = np.zeros_like(patch)
    # index = 2 if output_argmax else 0
    zeros[:, :, output_argmax] = 255
    
    return zeros

def segment_image(image, M, N, queries, return_sep = False):
    # Tiles for segmentation, recomended: word-shaped
    tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
    indices = [(x, x+M, y, y+N) for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)]
    
    # CLIP SetUp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(queries).to(device)
    
    canvas = np.zeros_like(image)
    
    for (x, w, y, h), tile in (zip(indices, tiles)):
        with torch.no_grad():
            tile_pil = Image.fromarray(tile)
            image_input = preprocess(tile_pil).unsqueeze(0).to(device)

            logits_per_image, logits_per_text = model(image_input, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            visual_label = create_visual_label(tile, probs.argmax()) * probs.max()
            canvas[x:w, y:h, :] = visual_label
        
    canvas_uint = canvas.astype(np.uint8) / 255
    if return_sep: return canvas_uint, image/255
    
    masked = canvas_uint * 0.5 + (image / 255) * 0.5
    
    return masked



def draw_polygons(image, polygons, color=(255, 0, 0), thickness=2):
    """
    Draws polygons on the given image.

    Parameters:
    image (numpy.ndarray): The input image on which polygons are to be drawn.
    polygons (list of list of list of int): A list of polygons, where each polygon is represented
                                            as a list of points, and each point is a list of two integers [x, y].
    color (tuple of int, optional): Color of the polygon lines in BGR format. Default is green (0, 255, 0).
    thickness (int, optional): Thickness of the polygon lines. Default is 2.

    Returns:
    numpy.ndarray: The image with polygons drawn on it.
    """
    # Make a copy of the image to avoid modifying the original one
    output_image = image.copy()

    # Iterate over each polygon
    for polygon in polygons:
        # Convert the list of points to a numpy array of shape (n, 1, 2), required by polylines
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        # Draw the polygon on the image
        cv.drawContours(output_image, [pts], -1, color, thickness)

    return output_image



def extract_bounding_box(polygon):
    """
    Extracts the bounding box (x, y, w, h) from a given polygon.

    Parameters:
    polygon (list of list of int): A list of points representing the polygon.

    Returns:
    tuple: A tuple containing (x, y, w, h) which represents the bounding box.
    """
    # Convert the polygon points to a numpy array for easier manipulation
    pts = np.array(polygon)

    # Extract x and y coordinates
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    # Calculate the minimum and maximum x and y values
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Calculate the width and height
    w = x_max - x_min
    h = y_max - y_min

    return x_min, y_min, w, h


def generate_polygon(x, y, w, h):
    """
    Generates a polygon (rectangle) given the bounding box parameters (x, y, w, h).

    Parameters:
    x (int): The x-coordinate of the top-left corner of the bounding box.
    y (int): The y-coordinate of the top-left corner of the bounding box.
    w (int): The width of the bounding box.
    h (int): The height of the bounding box.

    Returns:
    list of list of int: A list of points representing the polygon (rectangle).
    """
    # Define the four corners of the rectangle
    polygon = [
        [x, y],         # Top-left corner
        [x + w, y],     # Top-right corner
        [x + w, y + h], # Bottom-right corner
        [x, y + h]      # Bottom-left corner
    ]
    
    return polygon




def calculate_center(box):

    """
    Calculates the center coordinates of a bounding box.

    This function takes a bounding box defined by its top-left corner (x, y), width (w), 
    and height (h), and returns the coordinates of its center.

    Args:
        box (Tuple[int, int, int, int]): A tuple representing the bounding box 
                                         (x, y, width, height).

    Returns:
        Tuple[float, float]: The (x, y) coordinates of the center of the bounding box.
    """

    x, y, w, h = box
    return (x + w / 2, y + h / 2)

def are_horizontally_aligned(box1, box2, confidence_interval):

    """
    Checks if two bounding boxes are horizontally aligned within a specified confidence interval.

    This function determines if the vertical centers of two bounding boxes are within a 
    given confidence interval, indicating horizontal alignment.

    Args:
        box1 (Tuple[int, int, int, int]): The first bounding box (x, y, width, height).
        box2 (Tuple[int, int, int, int]): The second bounding box (x, y, width, height).
        confidence_interval (int): The maximum allowed vertical distance between the 
                                   centers of the boxes for them to be considered aligned.

    Returns:
        bool: True if the bounding boxes are horizontally aligned within the confidence interval, 
              False otherwise.
    """
    _, y1_center = calculate_center(box1)
    _, y2_center = calculate_center(box2)
    return abs(y1_center - y2_center) <= confidence_interval

def merge_boxes(box1, box2):

    """
    Merges two bounding boxes into one encompassing bounding box.

    This function takes two bounding boxes and returns a new bounding box that 
    encompasses both of them.

    Args:
        box1 (Tuple[int, int, int, int]): The first bounding box (x, y, width, height).
        box2 (Tuple[int, int, int, int]): The second bounding box (x, y, width, height).

    Returns:
        Tuple[int, int, int, int]: A new bounding box that encompasses both input boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    new_x = min(x1, x2)
    new_y = min(y1, y2)
    new_w = max(x1 + w1, x2 + w2) - new_x
    new_h = max(y1 + h1, y2 + h2) - new_y
    return (new_x, new_y, new_w, new_h)

def merge_horizontally_aligned_bounding_boxes(boxes, confidence_interval):
    """
    Merges horizontally aligned bounding boxes within a specified confidence interval.

    This function takes a list of bounding boxes and merges those that are horizontally 
    aligned within a given confidence interval. The process continues iteratively until 
    no more boxes can be merged.

    Args:
        boxes (List[np.ndarray]): List of bounding boxes to be merged. Each bounding box 
                                  is represented as a numpy array.
        confidence_interval (int): The maximum allowed vertical distance between boxes 
                                   for them to be considered horizontally aligned.

    Returns:
        List[np.ndarray]: A list of merged bounding boxes.
    """
    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip_indices = set()
        
        for i in range(len(boxes)):
            if i in skip_indices:
                continue
            box1 = boxes[i]
            merged_box = box1
            
            for j in range(i + 1, len(boxes)):
                if j in skip_indices:
                    continue
                box2 = boxes[j]
                
                if are_horizontally_aligned(merged_box, box2, confidence_interval):
                    merged_box = merge_boxes(merged_box, box2)
                    skip_indices.add(j)
                    merged = True
            
            new_boxes.append(merged_box)
        
        boxes = new_boxes
    
    return boxes



def get_lines_from_bboxes(ordered_merged_boxes:List[Tuple[int, ...]], image_max_width:int)-> List[Tuple[int, ...]]:

    """
    Generates horizontal line regions from a list of bounding boxes.

    This function takes a list of bounding boxes that are ordered and merged, 
    and generates horizontal line regions based on their vertical positions. 
    Each line region is represented by a tuple indicating its coordinates and size.

    Args:
        ordered_merged_boxes (List[Tuple[int, ...]]): A list of ordered and merged bounding boxes, 
                                                      where each bounding box is a tuple 
                                                      (x, y, width, height).
        image_max_width (int): The maximum width of the image, used to define the horizontal 
                               extent of the lines.

    Returns:
        List[Tuple[int, ...]]: A list of tuples representing the horizontal lines. Each tuple 
                               contains four integers (x, y, width, height) indicating the 
                               coordinates and size of the line region.
    """
     
    lines = []
    for bbox_idx in range(len(ordered_merged_boxes)-1):
        x_act, y_act, w_act, h_act =  ordered_merged_boxes[bbox_idx]
        x_nex, y_next, w_next, h_next = ordered_merged_boxes[bbox_idx+1]

        bottom_line = y_act + h_act
        next_top_line = y_next
        
        if bbox_idx == 0 and (abs(bottom_line - next_top_line) > 10):
            continue


        if bbox_idx == 0 and (abs(x_act - x_nex) > 50):
            continue
        

        lines.append([0, y_act , image_max_width, h_act])

        if (bbox_idx+1) == (len(ordered_merged_boxes) -1) and  (abs(bottom_line - next_top_line) < 30):
            lines.append([0, y_next , image_max_width, h_act])

    return lines


def extract_mean_std_from_consecutive_bboxes(bboxes:List[Tuple[int, ...]]):

    """
    Calculates the mean and standard deviation of the vertical distances between consecutive bounding boxes.

    This function takes a list of bounding boxes, computes the vertical distances between each consecutive 
    pair, and returns the mean and standard deviation of these distances. Each bounding box is represented 
    as a tuple of four integers: (x, y, width, height).

    Args:
        bboxes (List[Tuple[int, ...]]): List of bounding boxes, where each bounding box is a tuple 
                                        (x, y, width, height).

    Returns:
        Tuple[float, float]: A tuple containing:
                             - mean_consecutive_distances (float): The mean of the vertical distances 
                               between consecutive bounding boxes.
                             - std_consecutive_distances (float): The standard deviation of the vertical 
                               distances between consecutive bounding boxes.
    """

    mean_distances_bbox = {}

    for bbox_idx in range(len(bboxes) - 1):
        x_act, y_act, w_act, h_act =  bboxes[bbox_idx]
        x_nex, y_next, w_next, h_next = bboxes[bbox_idx+1]

        a = [x_act, y_act, w_act, h_act]
        b = [x_nex, y_next, w_next, h_next]

        bottom_line = y_act + h_act
        next_top_line = y_next
        mean_distances_bbox[(str(a), str(b))] = abs(bottom_line - next_top_line)

    mean_consecutive_distances = np.mean(list(mean_distances_bbox.values()))
    std_consecutive_distances = np.std(list(mean_distances_bbox.values()))

    return mean_distances_bbox, mean_consecutive_distances, std_consecutive_distances