#!/usr/bin/env python
import sys
import os

# Add the current directory to the path to import utils
sys.path.insert(0, os.path.dirname(__file__))

from utils import patch_gpu_max_mem_dynamically

# Apply the patch
patch_gpu_max_mem_dynamically()

# Now import and run the heareval embeddings runner
from heareval.embeddings.runner import runner

# Set sys.argv to mimic the original command-line call for click
# Original: python -m heareval.embeddings.runner [args]
# We want sys.argv to be ['heareval.embeddings.runner'] + arguments
sys.argv = ['heareval.embeddings.runner'] + sys.argv[1:]

# Call the runner function
runner()
