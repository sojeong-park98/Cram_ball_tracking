# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

cram = {
    'num_classes': 4,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38],
    'min_dim': 300,
    'steps': [8],
    'min_sizes': [15],
    'max_sizes': [50],
    'aspect_ratios': [],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CRAM',
}
