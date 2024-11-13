from random import random

def random_score(layers):
    return sorted(layers, key=lambda _: random())

def last_layer(layers):
    return layers[::-1]

def baseline(layers):
    return(['']*len(layers))

def one_layer(layers):
    pass