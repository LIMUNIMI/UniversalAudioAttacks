#!/usr/bin/env python
# coding: utf-8

# # Step 2.  
# ### Resample audio files to target sampling rates.
# Each model operates with audio files at a specific sr.  
# Resampling is required for embedding computation.

# In[6]:


import os
from pydub import AudioSegment

def resample(
    input_directory: str,
    target_sample_rates: list,
    split: str,
    n_folds: int = None
):
    """ Creates a new directory with the resampled audio files for each target sampling rate.
    
    Parameters
    ------------------------
    input_directory: str
        The audio files directory
    target_sample_rates: list
        The target sampling rates
    split: str
        The kind of split used (TVT, folds)
    n_folds: int = None
        The number of folds if split="folds" (optional)"""

    for target_sr in target_sample_rates:
        output_directory = os.path.join(input_directory + "/..", str(target_sr))
        if not(os.path.exists(output_directory)):
            print(f"Resampling to {target_sr}")
            print(f"Creating output directory {output_directory}")
            os.makedirs(output_directory)

            if split == "TVT":
                subfolders = ["test", "train", "valid"]

                for subfolder in subfolders:
                    os.makedirs(os.path.join(output_directory, subfolder))
                    audio_files = os.listdir(os.path.join(input_directory, subfolder))

                    for audio_file in audio_files:
                        if audio_file.endswith(".wav"):
                            sound = AudioSegment.from_wav(os.path.join(input_directory, subfolder, audio_file))
                            sound_new_sr = sound.set_frame_rate(target_sr)
            
                            output_path = os.path.join(output_directory, subfolder, audio_file)
                            sound_new_sr.export(output_path, format="wav")

            if split == "folds":
                subfolders = []
                for fold in range(n_folds):
                    subfolders.append("fold0" + str(fold))

                for subfolder in subfolders:
                    os.makedirs(os.path.join(output_directory, subfolder))
                    audio_files = os.listdir(os.path.join(input_directory, subfolder))

                    for audio_file in audio_files:
                        if audio_file.endswith(".wav"):
                            sound = AudioSegment.from_wav(os.path.join(input_directory, subfolder, audio_file))
                            sound_new_sr = sound.set_frame_rate(target_sr)
            
                            output_path = os.path.join(output_directory, subfolder, audio_file)
                            sound_new_sr.export(output_path, format="wav")
                            
        else:
            print(f"{output_directory} already exists")


# In[13]:


import json

# Read datasets information from JSON
with open('datasets.json', 'r') as file:
    datasets = json.load(file)

for task in datasets:
    # Get metadata
    task_name = task["name"]
    task_target_sr = task["target_sr"]

    metadata_path = f"tasks/{task_name}/task_metadata.json"
    
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    if metadata["split_mode"] == "new_split_kfold" or metadata["split_mode"] == "presplit_kfold":
        split = "folds"
        n_folds = metadata["nfolds"]
    elif metadata["split_mode"] == "trainvaltest":
        split = "TVT"
        n_folds = None

    # Resample
    resample(input_directory=f"tasks/{task_name}/48000",
            target_sample_rates=task_target_sr,
            split=split,
            n_folds=n_folds)


# In[14]:


# Additional count check (n. original audio files == n. resampled audio files)
def count_files_in_subfolders(root_folder):
    n_files = []
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            n_files.append(len(files))

    return n_files

# Read datasets information from JSON
with open('datasets.json', 'r') as file:
    datasets = json.load(file)

for task in datasets:
    # Get metadata
    task_name = task["name"]
    task_target_sr = task["target_sr"]

    for target_sr in task_target_sr:
        original_data_path = f"tasks/{task_name}/48000"
        resampled_data_path = f"tasks/{task_name}/{target_sr}"
    
        # Check if all audio files were resampled
        original_data_len = count_files_in_subfolders(original_data_path)
        resampled_data_len = count_files_in_subfolders(resampled_data_path)
        if original_data_len != resampled_data_len:
            print(f"Warning in {task_name} resample: original data length {original_data_len} - resampled data length {resampled_data_len}")
        else:
            print(f"OK - Correct resampling for {task_name} - {target_sr}")


# In[ ]:




