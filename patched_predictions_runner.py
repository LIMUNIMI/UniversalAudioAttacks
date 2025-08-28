#!/usr/bin/env python
import sys
import os

# Patch torch serialization to allowlist all classes from torch.nn.modules
import torch.serialization
from torch import nn, optim
import inspect

# List to hold all classes
all_classes = []

# Get classes from torch
for name, obj in inspect.getmembers(nn):
    if inspect.isclass(obj):
        all_classes.append(obj)

for name, obj in inspect.getmembers(nn.init):
    if inspect.isfunction(obj):
        all_classes.append(obj)

for name, obj in inspect.getmembers(optim):
    if inspect.isclass(obj):
        all_classes.append(obj)

# Add all classes to safe globals
torch.serialization.add_safe_globals(all_classes)

# Add the current directory to the path to import utils
sys.path.insert(0, os.path.dirname(__file__))

# patch nvmlGetDeviceName
from utils import patch_gpu_max_mem_dynamically

# Apply the patch
patch_gpu_max_mem_dynamically()

# Now import and run the heareval predictions runner
from heareval.predictions.runner import runner

# Set sys.argv to mimic the original command-line call for click
# Original: python -m heareval.predictions.runner [args]
sys.argv = ["heareval.predictions.runner"] + sys.argv[1:]

# Call the runner function
runner()
