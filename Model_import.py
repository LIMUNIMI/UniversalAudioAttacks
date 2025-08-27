#!/usr/bin/env python
# coding: utf-8

# # Step 3.
# #### Model import, embedding computation and evaluation
# Models are imported via pip + wget (for weights if present).
#
# **hearPipeline (hearPipeline2) env**
#
# **Do this via shell. Clone env before for safety.**
# *conda run pip install git+{git_url}.git@{git_rev_2021_12_01} {pip3_additional_packages}
# wget {zenodo_weights_url}*
#
# Embeddings are computed with:
#
# *import MODEL*
# *!python -m heareval.embeddings.runner MODEL --model WEIGHTS --tasks-dir ./tasks/ --embeddings-dir embeddings*
#
# Embeddings are evaluated (+ save MLP) with:
#
# *!python3 -m heareval.predictions.runner embeddings/MODEL/*
#
# /embeddings/MODEL/TASK/test.predicted-scores.json  contains results
# /savedModels/MODEL/TASK contains models (1 if single test split, k if k folds)

# #### Reference Example

# In[1]:


## Sample model import (for reference)

##   !conda run pip install git+https://github.com/hearbenchmark/hear-baseline.git@4478f9fd0d6cbc47fd06c66203b0340d1b5da1ad transformers==4.16.1 --no-deps
##   !conda run pip install git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU@7b7ce730d23232cec85698728fd1048800764d06

import os
import sys
from datetime import datetime
from utils import Tee, run_command

# Get script name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create directories for logs and images
log_dir = "logs"
image_dir = "images"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Redirect stdout and stderr to a log file and terminal
log_file = open(f"{log_dir}/{script_name}_{timestamp}.log", "w")
tee = Tee(sys.stdout, log_file)
sys.stdout = tee
sys.stderr = tee


# Ensure log file is closed on exit
def close_log():
    log_file.close()


import atexit

atexit.register(close_log)


# In[ ]:


## Sample compute embeddings (for reference)
# Note: to re run make sure the embedding directory is deleted.

##   import hearbaseline.wav2vec2
##   !python -m heareval.embeddings.runner hearbaseline.wav2vec2 --tasks-dir ./tasks/ --embeddings-dir embeddings

##   import GURA.fusion_hubert_xlarge
##   !python -m heareval.embeddings.runner GURA.fusion_hubert_xlarge --tasks-dir ./tasks/ --embeddings-dir embeddings


# In[ ]:


## Sample evaluate embeddings + save MLP (for reference)
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
# (!) Speech commands 5h and speech commands full have the same name and MLP are overwritten.

##   %env CUBLAS_WORKSPACE_CONFIG=:4096:8

##   # Train and evaluate classifier using hearbaseline.wav2vec2 embeddings
##   !python3 -m heareval.predictions.runner embeddings/hearbaseline.wav2vec2/*


# In[ ]:


## Sample for getting results (for reference)

##   import json
##   beijing_results = json.load(open("embeddings/hearbaseline.wav2vec2/beijing_opera/test.predicted-scores.json"))
##   beijing_results['aggregated_scores'] #['test_score_mean']


# In[1]:


# Removed IPython help command


# #### Wav2Vec

# In[2]:


MODEL_NAME = "hearbaseline.wav2vec2"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/hearbenchmark/hear-baseline.git@4478f9fd0d6cbc47fd06c66203b0340d1b5da1ad transformers==4.16.1 --no-deps


# In[2]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import hearbaseline.wav2vec2

run_command(
    "python patched_runner.py hearbaseline.wav2vec2 --tasks-dir ./tasks/ --embeddings-dir embeddings"
)


# In[3]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train and evaluate classifier using hearbaseline.wav2vec2 embeddings + save MLP
run_command(
    "python3 -m heareval.predictions.runner embeddings/hearbaseline.wav2vec2/* "
)


# In[4]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[5]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[6]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Esc50 0.5610 -> 0.6144   UP (!)
# Gunshot 0.8482 -> 0.7708   DOWN (!)
# Libricount 0.6921 -> 0.6642   DOWN (!)
# Mridangam stroke 0.9432 -> 0.9190   DOWN (!)
# Mridangam tonic 0.8283 -> 0.7549   DOWN (!)
# NsyntPitch5h 0.4020 -> 0.4360   UP
# Speech commands 0.8382 -> 0.8801   UP (!)
# Voxlingua 0.4928 -> 0.5576   UP (!)
# GTZAN_music_speech 0.9462 -> 0.9288   DOWN
# Crema-D 0.6562 -> 0.6796   UP


# #### GURA Hubert Fusion

# In[1]:


MODEL_NAME = "GURA.fusion_hubert_xlarge"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU@7b7ce730d23232cec85698728fd1048800764d06


# In[1]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import GURA.fusion_hubert_xlarge

run_command(
    "python patched_runner.py GURA.fusion_hubert_xlarge --tasks-dir ./tasks/ --embeddings-dir embeddings"
)


# In[2]:


# For crema-D (added later)
# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import GURA.fusion_hubert_xlarge

run_command(
    "python patched_runner.py GURA.fusion_hubert_xlarge --tasks-dir ./tasks/ --embeddings-dir embeddings"
)


# In[3]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on GURA.fusion_hubert_xlarge embeddings + save MLP
run_command(
    "python3 -m heareval.predictions.runner embeddings/GURA.fusion_hubert_xlarge/*  "
)


# In[4]:


# For crema-D (added later)
# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on GURA.fusion_hubert_xlarge embeddings + save MLP
run_command(
    "python3 -m heareval.predictions.runner embeddings/GURA.fusion_hubert_xlarge/*  "
)


# In[2]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[5]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[9]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Gunshot 0.9286 -> 0.9137   DOWN
# NsyntPitch5h 0.3820 -> 0.4000   UP
# GTZAN_music 0.7960 -> 0.8189   UP


# #### RedRice EfficientNet

# In[1]:


MODEL_NAME = "efficient_latent"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/RicherMans/HEAR2021_EfficientLatent@0ec444d99f9e3d6c7dc95cb715dfa249d516e58a
##   wget https://zenodo.org/record/6332525/files/hear2021-efficient_latent.pt


# In[13]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import efficient_latent

run_command(
    "python patched_runner.py efficient_latent --tasks-dir ./tasks/ --embeddings-dir embeddings --model ./modelWeights/hear2021-efficient_latent.pt"
)


# In[15]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on efficient_latent embeddings + save MLP
run_command("python3 -m heareval.predictions.runner embeddings/efficient_latent/*  ")


# In[2]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[3]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[4]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results, print results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed

# Gunshot 0.8780 -> 0.9285    UP (!)
# speech_commands_5h 0.5734 -> 0.5926   UP
# GTZAN_music_speech 0.9679 -> 0.9916   UP
# Crema-D 0.5746 -> 0.5532   DOWN


# #### GURA Wav2Vec Fusion

# In[2]:


MODEL_NAME = "GURA.fusion_wav2vec2"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU@7b7ce730d23232cec85698728fd1048800764d06


# In[3]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import GURA.fusion_wav2vec2

run_command(
    "python patched_runner.py GURA.fusion_wav2vec2 --tasks-dir ./tasks/ --embeddings-dir embeddings"
)


# In[4]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on GURA.fusion_wav2vec2 embeddings + save MLP
run_command(
    "python3 -m heareval.predictions.runner embeddings/GURA.fusion_wav2vec2/*  "
)


# In[3]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[4]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[5]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Gunshot 0.9673 -> 0.9285    DOWN
# Libricount 0.6526 -> 0.6351  DOWN
# NSynth_pitch_5h 0.3300 -> 0.3440   UP
# GTZAN_music_speech 0.9532 -> 0.9371   DOWN
# Crema-D 0.6924 -> 0.7074   UP


# #### GURA Fuse Cat H+w+C

# In[1]:


MODEL_NAME = "GURA.fusion_cat_xwc"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU@7b7ce730d23232cec85698728fd1048800764d06


# In[9]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import GURA.fusion_cat_xwc

run_command(
    "python patched_runner.py GURA.fusion_cat_xwc --tasks-dir ./tasks/ --embeddings-dir embeddings"
)


# In[10]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on GURA.fusion_cat_xwc embeddings + save MLP
run_command("python3 -m heareval.predictions.runner embeddings/GURA.fusion_cat_xwc/*  ")


# In[2]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[3]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[4]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[5]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Beijing_opera 0.9660 -> 0.9534   DOWN
# Esc50 0.7335 -> 0.7440   UP
# Gunshot 0.9345 -> 0.9136   DOWN
# NSynth_pitch_5h 0.8460 -> 0.8339   DOWN
# GTZAN_music 0.8050 -> 0.8170   UP
# GTZAN_music_speech 0.9282 -> 0.9436   UP


# #### GURA Avg H+w+C

# In[3]:


MODEL_NAME = "GURA.avg_xwc"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU@7b7ce730d23232cec85698728fd1048800764d06


# In[2]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import GURA.avg_xwc

run_command(
    "python patched_runner.py GURA.avg_xwc --tasks-dir ./tasks/ --embeddings-dir embeddings"
)


# In[3]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on GURA.avg_xwc embeddings + save MLP
run_command("python3 -m heareval.predictions.runner embeddings/GURA.avg_xwc/*  ")


# In[4]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[5]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[6]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Beijing_opera 0.9448 -> 0.9278   DOWN
# Gunshot 0.8571 -> 0.6845   DOWN (!)
# Crema-D 0.5473 -> 0.5284   DOWN


# #### CVSSP PANNS

# In[1]:


MODEL_NAME = "panns_hear"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/qiuqiangkong/HEAR2021_Challenge_PANNs@daae61a072d0102ef224e5c7c4038bf5960c43c5
##   !wget https://zenodo.org/record/6332525/files/hear2021-panns_hear.pth


# In[8]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import panns_hear

run_command(
    "python patched_runner.py panns_hear --tasks-dir ./tasks/ --embeddings-dir embeddings --model ./modelWeights/hear2021-panns_hear.pth"
)


# In[6]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on panns_hear embeddings + save MLP
run_command("python3 -m heareval.predictions.runner embeddings/panns_hear/*  ")


# In[2]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[3]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[4]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Beijing_opera 0.9112 -> 0.9237   UP
# Gunshot 0.7976 -> 0.8571   UP
# Mridangam_stroke 0.9391 -> 0.9276   DOWN
# NSynth_pitch_5h 0.1480 -> 0.1299   DOWN


# #### CP-JKU PaSST base

# In[1]:


MODEL_NAME = "hear21passt.base"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[2]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/kkoutini/passt_hear21@5487d2f4b7782e2879dbf6fd0b1135a5b137b106
##   !wget https://zenodo.org/record/6332525/files/hear2021-hear21passt.base.pt


# In[5]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import hear21passt.base

run_command(
    "python patched_runner.py hear21passt.base --tasks-dir ./tasks/ --embeddings-dir embeddings --model ./modelWeights/hear2021-hear21passt.base.pt"
)


# In[6]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on hear21passt.base embeddings + save MLP
run_command("python3 -m heareval.predictions.runner embeddings/hear21passt.base/*  ")


# In[2]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[3]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[4]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Beijing_opera 0.9660 -> 0.9536   DOWN
# Gunshot 0.9405 -> 0.8959   DONW (!)
# Libricount 0.6601 -> 0.6329   DOWN
# Mridangam_tonic 0.8194 -> 0.8046   DOWN
# NSynth_pitch_5h 0.2560 -> 0.2300   DOWN
# Speech_commands_5h 0.6810 -> 0.6928   UP
# Voxlingua 0.2593 -> 0.2439   DOWN
# GTZAN_music_speech 0.9769 -> 0.9615   DOWN


# #### Stellenbosch LSL DBERT

# In[1]:


MODEL_NAME = "audio_dbert"

import os
import json

with open("datasets.json", "r") as file:
    datasets = json.load(file)


# In[ ]:


# Model import (on shell)
##   !conda run pip install git+https://github.com/RF5/audio_dbert@b2d0e44de789ff340711730425de1df0dba091da
##   wget https://zenodo.org/record/6332525/files/hear2021-audio_dbert.pt


# In[3]:


# Compute embeddings
# Note: to re run make sure the embedding directory is deleted.
import audio_dbert

run_command(
    "python patched_runner.py audio_dbert --tasks-dir ./tasks/ --embeddings-dir embeddings --model ./modelWeights/hear2021-audio_dbert.pt"
)


# In[2]:


# Check if embeddings were computed for all tasks
embeddings_count = 0
for task in datasets:
    task_name = task["name"]
    embedding_path = f"embeddings/{MODEL_NAME}/{task_name}"
    if os.path.exists(embedding_path):
        embeddings_count += 1

if embeddings_count == len(datasets):
    print(f"OK - Embeddings computed for all tasks with model {MODEL_NAME}")
else:
    print(f"Missing some embeddings: expected {len(datasets)} found {embeddings_count}")


# In[ ]:


# Evaluate embeddings + save MLP
# Note: to re run make sure the predictions files in the task embedding directory and the saved models are deleted.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Train, evaluate and save MLP classifier/s on audio_dbert embeddings + save MLP
run_command("python3 -m heareval.predictions.runner embeddings/audio_dbert/*  ")


# In[3]:


# Rename saved models folders and files
with open("savedModelsName.json", "r") as file:
    models_name = json.load(file)

models_count = 0
for task_name in models_name:
    old_name = task_name["old_name"]
    new_name = task_name["name"]
    old_path = f"savedModels/{MODEL_NAME}/{old_name}"

    # Rename folder
    new_path = f"savedModels/{MODEL_NAME}/{new_name}"
    if os.path.exists(old_path):
        print(f"Renaming folder {old_name} to {new_name}")
        os.system(f"mv {old_path} {new_path}")

    # Rename files
    for old_file_name in os.listdir(new_path):
        if old_name in old_file_name:
            new_file_name = old_file_name.replace(old_name, new_name)
            print(f"Renaming file {old_file_name} to {new_file_name}")
            os.system(f"mv {new_path}/{old_file_name} {new_path}/{new_file_name}")


# In[4]:


# Check if MLP models were saved for all tasks
models_count = 0
for task in datasets:
    task_name = task["name"]

    models_path = f"savedModels/{MODEL_NAME}/{task_name}"
    if os.path.exists(models_path):
        models_count += 1

if models_count == len(datasets):
    print(f"OK - MLP Models saved for all tasks for model {MODEL_NAME}")
else:
    print(f"Missing some MLP models: expected {len(datasets)} found {models_count}")


# In[5]:


# Get results of models on clean embeddings
model_clean_results = {}
for task in datasets:
    task_name = task["name"]

    results_path = f"embeddings/{MODEL_NAME}/{task_name}/test.predicted-scores.json"
    # Check if results were computed for all tasks
    if os.path.exists(results_path):
        # Get results
        clean_results = json.load(open(results_path))
        model_clean_results[task_name] = []
        model_clean_results[task_name].append(clean_results)
    else:
        print(f"Results not computed for task {task_name}")
print("")

# model_clean_results contains the results
print(f"-- {MODEL_NAME} -- Results:")
for task in model_clean_results:
    # Get task split
    embeddings_path = f"embeddings/{MODEL_NAME}/{task}"
    metadata_path = f"{embeddings_path}/task_metadata.json"

    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    if (
        metadata["split_mode"] == "new_split_kfold"
        or metadata["split_mode"] == "presplit_kfold"
    ):
        split = "folds"
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"

    print(f"- Test score for task {task}:")
    if split == "folds":
        print(model_clean_results[task][0]["aggregated_scores"]["test_score_mean"])
    elif split == "TVT":
        print(model_clean_results[task][0]["test"]["test_score"])

    print("")


# In[ ]:


# Notable variations (> 0.01) to HEAR Leaderboard -> computed
# Gunshot 0.8095 -> 0.8780   UP (!)


# #### Notes

# In[ ]:


# Task Metadata was wrong for mrindingam stroke and tonic and carried over to emb folder, now it is fixed in the tasks folder.
