import torch
from torch import nn
import re
from score_functions import *

def resnet_mask(model, num_layers_reinitialize = 0, score_function=random_score):
     # Get all layer names as a single string
    all_layers = [(a, b) for (a, b) in model.named_modules()]
    filtered_layers = []
    print('all layers:')
    print(all_layers)
    # Use regex to find all patterns like "layer1.0", "layer1.1", etc.
    layer_blocks = []
    current_block = []
    current_pattern = None

    for layer in all_layers:
        match = re.search(r'layer(\d+)\.(\d+)', layer[0])
        if match:
            pattern = f"layer{match.group(1)}.{match.group(2)}"
            if pattern != current_pattern:
                if current_block:
                    layer_blocks.append(current_block)
                current_block = []
                current_pattern = pattern
            current_block.append(layer)

    if current_block:
        layer_blocks.append(current_block)

    if score_function == one_layer:
        blocos = layer_blocks[num_layers_reinitialize-1:num_layers_reinitialize]
    else:
        blocos = score_function(layer_blocks)[:num_layers_reinitialize]

    # List of allowed module types instead of string suffixes
    allowed_types = [nn.Conv2d, nn.BatchNorm2d]

    # Filter the layers based on whether they start with `blocos` and are of an allowed type
    filtered_layers = list(
        filter(
            lambda layer: any(str(prefix[0][0]) in str(layer[0]) for prefix in blocos) and
                          isinstance(layer[1], tuple(allowed_types)),
            all_layers
        )
    )
    
    print(f"filtered layers: {filtered_layers}")
    return [x[0] for x in filtered_layers]
    
def vgg_mask(model, num_layers_reinitialize= 0, score_function=random_score):
    all_layers = [(a, b) for (a, b) in model.named_modules()]
    filtered_layers = []

    # Use regex to find all patterns like "features.1", "features.2", "features.14", etc.
    allowed_types = [nn.Conv2d, nn.BatchNorm2d]

    blocos = sorted(
        list(
            filter(
                lambda layer: re.search(r'features\.\d+', layer[0]) and layer[1].type not in allowed_types, 
                all_layers
            )
        )
        ,key=lambda x: int(re.search(r'features\.(\d+)', x[0]).group(1))
    )

    print(blocos)
    blocos = score_function(blocos)[:num_layers_reinitialize]

    # List of allowed module types instead of string suffixes
    aasd = [int(x[0].split('.')[1]) for x in blocos]
    print(aasd)
    print(sorted(aasd))

    return list([x[0] for x in blocos])


    # Filter the layers based on whether they start with `blocos` and are of an allowed type
    filtered_layers = list(
        filter(
            lambda layer: any(layer[0].startswith(prefix[0]) for prefix in blocos) and
                          isinstance(layer[1], tuple(allowed_types)),
            all_layers
        )
    )
    
    #print(filtered_layers)
    #return [x[0] for x in filtered_layers]


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


def mobilenet_v2_mask(model, num_layers_reinitialize= 1, score_function=random_score):    
    all_layers = [(a, b) for (a, b) in model.named_modules()]
    filtered_layers = []

    # Use regex to find all patterns like "features.13.conv.0.0", "features.13.conv.0.1"

    def layer_key(layer):
        parts = layer[0].split('.')
        return tuple(int(part) if part.isdigit() else part for part in parts)

    blocos = sorted(
        set(
            filter(
                lambda layer: re.search(r'features\.\d+\.conv\.\d+\.\d+', layer[0]), 
                all_layers
            )
        ),
        key=layer_key
    )
    print(blocos)
    blocos = score_function(blocos)[:num_layers_reinitialize]
    print(blocos)

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

def layer_to_reinitizalize(model, num_layers_reinitialize = 0, architecture="resnet20", score_function=random_score):
    if architecture.startswith("resnet"):
        return resnet_mask(model, num_layers_reinitialize, score_function)
    elif architecture.startswith("vgg"):
        return vgg_mask(model, num_layers_reinitialize, score_function)
    elif architecture == 'mobilenet_v2':
        return mobilenet_v2_mask(model, num_layers_reinitialize, score_function)
    else:
        pass

