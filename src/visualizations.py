

import torch
import torch.nn as nn

from  torchtyping import TensorType

import os
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import scipy.stats as st 


from typing import *

import pickle
import pdb

import seaborn as sns



from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_cols = len(imgs)
    num_rows = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for col_idx, row in enumerate(imgs):
        for row_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    #plt.show()
    plt.savefig("dummy_example_batch.png")


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_violin_plot_from_freq_attribute_distances(specific_attribute_embeddings,
                                                   dic_realtion_names:dict,
                                                   file_path:str):
    n = specific_attribute_embeddings.shape[0]
    distances = torch.cdist(specific_attribute_embeddings, specific_attribute_embeddings, p=2)#torch.sum((specific_attribute_embeddings - specific_attribute_embeddings)**2, dim=1).sqrt().numpy()
    distances = distances.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1) 
    
    mean_distances_attributes = torch.mean(distances).item()
    std_distance_attributes = torch.std(distances).item()
    
    
    labels = []
    data = []
    for label, indexes in dic_realtion_names:
        distance = torch.cdist(specific_attribute_embeddings[indexes], specific_attribute_embeddings[indexes], p=2)
        n = distance.shape[0]
        distance = distance.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1).flatten().numpy()

        labels.append(label)
        data.append(distance)
    
    
        
    parts = plt.violinplot(data, showmeans=False, showmedians=False,
        showextrema=False)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    


    plt.xticks(range(1, len(labels) +1), labels, rotation=45)

    plt.axhline(mean_distances_attributes, color='#CC4F1B', linestyle = '--', label="Global Avg Distance Mean")
    plt.axhspan(mean_distances_attributes-std_distance_attributes, mean_distances_attributes+std_distance_attributes, edgecolor='#CC4F1B', facecolor='#FF9848', alpha=0.5)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(file_path)

    
    plt.close()
    
    
def plot_attribute_metric_space(attribute_embedding_space: TensorType["population", "nAttributes", "embedding_size"],
                                fig_name:str):
    
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)

    population, n_attributes, embedding_size = attribute_embedding_space.shape

    # Flatten the embedding space to [total attributes, embedding size]
    flattened_embeddings = attribute_embedding_space.reshape(population * n_attributes, embedding_size)

    # Generate labels for the attribute types
    labels = np.tile(np.arange(n_attributes), population)

    # Apply UMAP for dimensionality reduction (reduce to 2D for plotting)
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(flattened_embeddings)

    # Create a scatter plot of the reduced embeddings
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='Accent', s=15)

    # Create a legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Attribute {i}',
                              markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=8) for i in range(n_attributes)]
    plt.legend(handles=legend_elements, title="Attribute Types")

    # Add title and labels
    plt.title("UMAP of Attribute Embedding Space")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    plt.savefig(fig_name)
    plt.close()

    # Show the plot
    
    
    
def plot_positional_encoding(positional_encoding: TensorType["seq_length", "embedding_size"],
                             fig_name: str):
    """
    Plots a heatmap of the positional encoding learned by the model.

    Parameters:
    positional_encoding: Tensor of shape [seq_length, embedding_size]
        The learned positional encodings for each position in the sequence.

    The function does not return anything, but it displays a heatmap of the positional encoding.
    """
    plt.figure(figsize=(12, 6))

    # Plot heatmap using seaborn
    print(positional_encoding.shape)
    sns.heatmap(positional_encoding, cmap='viridis', cbar=True)

    # Add title and labels
    plt.title("Positional Encoding Heatmap")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position in Sequence")

    # Show the plot
    plt.savefig(fig_name)

    plt.show()

    
def plot_embedding_distribution(embeddings, save_path:str):

    
    # Reduce dimensionality to 2D using UMAP
    reducer = umap.UMAP(n_components=2)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create a scatter plot for train embeddings
    plt.figure(figsize=(12, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    plt.title('Embeddings Distribution')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)

    plt.close()



def plot_train_test_embedding_distribution(train_embeddings, train_labels, test_embeddings, test_labels, map_entities, save_path='embedding_distributions.png'):
    """
    Plots the distribution of train and test embeddings with different colors for different labels using UMAP.

    Parameters:
    train_embeddings (np.ndarray): Embeddings for the training set.
    train_labels (list or np.ndarray): Labels for the training set.
    test_embeddings (np.ndarray): Embeddings for the test set.
    test_labels (list or np.ndarray): Labels for the test set.
    save_path (str): Path to save the plot.
    """
    
    # Reduce dimensionality to 2D using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    train_embeddings_2d = reducer.fit_transform(train_embeddings)
    test_embeddings_2d = reducer.transform(test_embeddings)
    
    # Create a scatter plot for train embeddings
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    unique_labels = np.unique(train_labels)
    for label in unique_labels:
        indices = np.where(train_labels == label)
        plt.scatter(train_embeddings_2d[indices, 0], train_embeddings_2d[indices, 1], label=f'Label {map_entities[label]}')
    plt.title('Train Embeddings Distribution')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    
    # Create a scatter plot for test embeddings
    plt.subplot(1, 2, 2)
    unique_labels = np.unique(test_labels)
    for label in unique_labels:
        
        indices = np.where(test_labels == label)
        plt.scatter(test_embeddings_2d[indices, 0], test_embeddings_2d[indices, 1], label=f'Label {map_entities[label]}')
    plt.title('Test Embeddings Distribution')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)

    plt.close()


def plot_shift(embedding_matrix_1, embedding_matrix_2, fig_path: str, labels=None, title='Embedding Shift with UMAP'):
    """
    Plots the shift of embeddings from two embedding matrices using UMAP for dimensionality reduction.
    
    :param embedding_matrix_1: numpy array of shape (n_samples, n_features), original embeddings
    :param embedding_matrix_2: numpy array of shape (n_samples, n_features), embeddings after training
    :param labels: List of labels for each point (optional)
    :param title: Title of the plot
    """
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    
    # Validate input shapes
    if embedding_matrix_1.shape != embedding_matrix_2.shape:
        raise ValueError("Both embedding matrices must have the same shape.")
    
    n_samples = embedding_matrix_1.shape[0]
    
    # Combine the embedding matrices
    combined_embeddings = np.vstack((embedding_matrix_1, embedding_matrix_2))
    
    # Initialize UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=10)
    
    # Fit UMAP on the combined data and transform
    embeddings_2d = reducer.fit_transform(combined_embeddings)
    
    # Split back into original and new
    embeddings_1_2d = embeddings_2d[:n_samples]
    embeddings_2_2d = embeddings_2d[n_samples:]
    
    plt.figure(figsize=(12, 10))
    plt.title(title, fontsize=16)
    
    # Plot embeddings and their shifts
    for i in range(n_samples):
        x1, y1 = embeddings_1_2d[i]
        x2, y2 = embeddings_2_2d[i]
        
        # Plot original position
        plt.scatter(x1, y1, color='blue', alpha=0.6, label='Original' if i == 0 else "")
        
        # Plot new position
        plt.scatter(x2, y2, color='red', alpha=0.6, label='New' if i == 0 else "")
        
        # Draw a line between the original and new position
        plt.plot([x1, x2], [y1, y2], color='gray', linestyle='--', linewidth=1)
        
        # Optionally add labels
        if labels is not None:
            plt.text(x1, y1, labels[i], fontsize=9, color='blue', alpha=0.7)
            plt.text(x2, y2, labels[i], fontsize=9, color='red', alpha=0.7)
    
    plt.xlabel('UMAP Component 1', fontsize=14)
    plt.ylabel('UMAP Component 2', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

