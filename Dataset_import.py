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
import sys
from datetime import datetime

# Get script name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create directories for logs and images
log_dir = "logs"
image_dir = "images"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Redirect stdout and stderr to a log file
log_file = open(f"{log_dir}/{script_name}_{timestamp}.log", "w")
sys.stdout = log_file
sys.stderr = log_file


# Ensure log file is closed on exit
def close_log():
    log_file.close()


import atexit

atexit.register(close_log)

# Read datasets information from JSON
with open("datasets.json", "r") as file:
    datasets = json.load(file)

# Perform downloads
for task in datasets:
    task_name = task["name"]
    task_url = task["url"]
    old_task_name = task["old_name"]

    Path("tasks_compressed").mkdir(exist_ok=True)

    if not (os.path.exists(f"tasks/{task_name}")):
        if not (os.path.exists(f"tasks_compressed/{task_name}")):
            print(f"Downloading: {task_name}")
            os.system(f"wget {task_url} -O tasks_compressed/{task_name}")
        else:
            print(f"Skipping download of {task_name} - already downloaded")

        print(f"Extracting and renaming: {task_name}")
        if not old_task_name.startswith("hear-2021"):
            old_task_name = "tasks/" + old_task_name
        os.system(
            f"tar -zxf tasks_compressed/{task_name} && mv {old_task_name} tasks/{task_name}"
        )
    else:
        print(f"{task_name} already prepared")


# In[ ]:
