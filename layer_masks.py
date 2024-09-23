import torch
from torch import nn
import re
from score_functions import random_score


def resnet_mask(model, p_reinitialized= .5, score_function=random_score):
     # Get all layer names as a single string
    all_layers = [(a, b) for (a, b) in model.named_modules()]
    filtered_layers = []

    # Use regex to find all patterns like "layer1.0", "layer1.1", etc.
    blocos = sorted(
        set(
            filter(
                lambda layer: re.search(r'layer\d+\.\d+', layer[0]), 
                all_layers
            )
        )
    )
    
    # Score and reduce the number of layers based on `p_reinitialized_layers`
    blocos = score_function(blocos)[:int(len(blocos) * p_reinitialized)]

    # List of allowed module types instead of string suffixes
    allowed_types = [nn.Conv2d, nn.BatchNorm2d]

    # Filter the layers based on whether they start with `blocos` and are of an allowed type
    filtered_layers = list(
        filter(
            lambda layer: any(layer[0].startswith(prefix[0]) for prefix in blocos) and
                          isinstance(layer[1], tuple(allowed_types)),
            all_layers
        )
    )
    
    print(filtered_layers)
    return [x[0] for x in filtered_layers]
    
def vgg_mask(model, p_reinitialized= .5, score_function=random_score):
    all_layers = [(a, b) for (a, b) in model.named_modules()]
    filtered_layers = []

    # Use regex to find all patterns like "features.1", "features.2", "features.14", etc.
    blocos = sorted(
        set(
            filter(
                lambda layer: re.search(r'features\.\d+', layer[0]), 
                all_layers
            )
        )
    )
    
    # List of allowed module types instead of string suffixes
    allowed_types = [nn.Conv2d, nn.BatchNorm2d]

    # Filter the layers based on whether they start with `blocos` and are of an allowed type
    filtered_layers = list(
        filter(
            lambda layer: any(layer[0].startswith(prefix[0]) for prefix in blocos) and
                          isinstance(layer[1], tuple(allowed_types)),
            all_layers
        )
    )
    
    print(filtered_layers)
    return [x[0] for x in filtered_layers]

def layer_to_reinitizalize(model, p_reinitialized= .5, architecture="resnet20", score_function=random_score):
    if architecture.startswith("resnet"):
        return resnet_mask(model, p_reinitialized, score_function)
    elif architecture == 'vgg':
        return vgg_mask(model, p_reinitialized, score_function)
    else:
        pass

