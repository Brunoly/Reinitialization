from torchvision.models import vit_b_32
model  = vit_b_32(weights=None)
all_layers = [(a, b) for (a, b) in model.named_modules()]
from pprint import pprint
pprint(all_layers)
