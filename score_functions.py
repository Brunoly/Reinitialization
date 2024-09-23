from random import random

def random_score(layers):
    scored_layers = [(layer, random()) for layer in layers]    
    return [layer for layer, score in sorted(scored_layers, key=lambda x: x[1])]

def last_layer(layers):
    scored_layers = [(layer, -i) for (i, layer) in enumerate(layers)]    
    return [layer for layer, score in sorted(scored_layers, key=lambda x: x[1])]