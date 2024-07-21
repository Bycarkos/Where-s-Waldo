import torch
import torch.nn as nn

from  torchtyping import TensorType



def evaluate_shift(x_previous:TensorType["nodes", "embedding_size", "numpy", "float"],
                   x_after:TensorType["nodes", "embedding_size", "numpy", "float"],
                   population: TensorType["batch", "", "numoy", "int32"],
                   fig_name: str):
    
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    class_previous = ["blue" for i in range(population.shape[-1])]
    class_actual = ["green" for i in range(population.shape[-1])]
    labels = class_previous + class_actual
    embeddings = np.vstack((x_previous[population], x_after[population]))
    
    mapper = umap.UMAP().fit(embeddings)
    
    
    coordinates = mapper.embedding_
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Plot the previous embeddings
    plt.scatter(coordinates[:population.shape[-1], 0], coordinates[:population.shape[-1], 1], color='blue', label='Previous', alpha=0.6)
    
    # Plot the after embeddings
    plt.scatter(coordinates[population.shape[-1]:, 0], coordinates[population.shape[-1]:, 1], color='red', label='After', alpha=0.6)
    
    # Draw lines connecting the previous and after embeddings
    for i in range(len(population)):
        plt.plot([coordinates[i, 0], coordinates[i+len(population), 0]],
                 [coordinates[i, 1], coordinates[i+len(population), 1]], 'gray', alpha=0.5)
    
    # Add legend
    plt.legend()
    
    # Add title and labels
    plt.title('Embedding Shift Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Save the figure
    plt.savefig(fig_name)
    plt.close()

    
    #umap.plot.points(mapper, labels=labels)



def contrastive_loss(x1, x2,  label, margin=1.0):
    """
    Compute the contrastive loss.

    Parameters:
    y_true (Tensor): Tensor of labels indicating whether pairs are similar or dissimilar (1 for similar, 0 for dissimilar)
    y_pred (Tensor): Tensor of predicted distances between pairs
    margin (float): Margin for dissimilar pairs. Default is 1.0

    Returns:
    Tensor: Contrastive loss
    """
    
    y_pred = torch.nn.functional.pairwise_distance(x1, x2)
    # Ensure y_true is of type float
    y_true = label.float()
    
    # Calculate positive loss (for similar pairs)
    pos_loss = label * torch.square(y_pred)
    
    # Calculate negative loss (for dissimilar pairs)
    neg_loss = (1 - label) * torch.square(torch.clamp(margin - y_pred, min=0.0))
    
    # Sum the losses
    loss = torch.mean(pos_loss + neg_loss)
    
    return loss

def compute_resize(shape, patch_size):
    offset = shape % patch_size
    if offset > (patch_size /2):
        return shape + (patch_size - offset)
    else:
        return shape - offset
    
    
def extract_optimal_shape(min_width:int,  max_height:int, patch_size:int):
    
    min_width_recovered = compute_resize(shape=min_width, patch_size=patch_size)
    max_height_recovered = compute_resize(shape=max_height, patch_size=patch_size)
    
    return min_width_recovered, max_height_recovered
    
    
    
