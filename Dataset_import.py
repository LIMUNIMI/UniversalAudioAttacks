#!/usr/bin/env python
# coding: utf-8

# # Step 1.  
# #### Dataset download, import and decompression
# The datasets are downloaded using wget and decompressed using tar.  
# The file datasets.json contains the url of all the task datasets and names.  
# The following loop iters over the json file to download all the datasets.  
# 
# 
# wget {url} -O NAME  
# tar -zxf NAME

# In[ ]:


import json
import os
from pathlib import Path

# Read datasets information from JSON
with open('datasets.json', 'r') as file:
    datasets = json.load(file)

# Perform downloads
for task in datasets:
    task_name = task["name"]
    task_url = task["url"]
    old_task_name = task["old_name"]

    Path("tasks_compressed").mkdir(exist_ok=True)

    if not(os.path.exists(f"tasks/{task_name}")):
        print(f"Downloading: {task_name}")
        
        # Download
        os.system(f"wget {task_url} -O tasks_compressed/{task_name}")
    
        # Extract and Rename
        os.system(f"tar -zxf tasks_compressed/{task_name} && mv tasks/{old_task_name} tasks/{task_name}")
    else:
        print(f"Skipping download of {task_name} - already downloaded")


# In[ ]:




