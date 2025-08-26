#!/usr/bin/env python
# coding: utf-8

# # Step 5.
# #### SVM-based detection of adversarial examples
# Train and evaluate SVM on binary task: benign vs adversarial embeddings to determine whether adversarial embeddings can be detected.
# 
# #### For each model, for each task
# * Get clean and adversarial embeddings
# * Define SVM classifier
# * Train and evaluate SVM classifier
# 
# **attackPipeline (attackHyb) Env**

# In[1]:


# Libraries
import json
import pickle
import numpy as np
import os
import pandas as pd
import sys
from datetime import datetime
from utils import Tee

# Get script name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create directories for logs and images
log_dir = "logs"
image_dir = "images"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Redirect stdout and stderr to a log file and terminal
log_file = open(f"{log_dir}/{script_name}_{timestamp}.log", 'w')
tee = Tee(sys.stdout, log_file)
sys.stdout = tee
sys.stderr = tee

# Ensure log file is closed on exit
def close_log():
    log_file.close()
import atexit
atexit.register(close_log)


# In[2]:


def get_embeddings(
    embeddings_path: str,
    split: str,
    n_folds: int = None
):
    """ Return the embeddings for the given embeddings path.
    
    Parameters
    ------------------------
    embeddings_path: str
        The embeddings path
    split: str
        The kind of split used (TVT, folds)
    n_folds: int = None
        The number of folds if split="folds" (optional)
        
    Returns
    ------------------------
    embeddings: dict[list[np.ndarray[np.float32]]. Array shape: (embedding_dim,)
        The dict with the embeddings organized per splits."""

    if split == "TVT":
        embeddings = {}
        embeddings["test"] = []
        
        embeddings_files_path = f"{embeddings_path}/test"
        for file in sorted(os.listdir(embeddings_files_path)):
            if file.endswith(".npy"):
                embedding_path = os.path.join(embeddings_files_path, file)
                embedding = np.load(embedding_path)
                embeddings["test"].append(embedding)

        print(f"stored test embeddings {embeddings_files_path}")

        return embeddings

    elif split == "folds":
        embeddings = {}
        
        for fold in range(n_folds):
            embeddings[f"fold0{fold}"] = []
            
            embeddings_files_path = f"{embeddings_path}/fold0{fold}"
            for file in sorted(os.listdir(embeddings_files_path)):
                if file.endswith(".npy"):
                    embedding_path = os.path.join(embeddings_files_path, file)
                    embedding = np.load(embedding_path)
                    embeddings[f"fold0{fold}"].append(embedding)
                    
            print(f"stored in fold0{fold} embeddings {embeddings_files_path}")

        return embeddings


# In[3]:


# Pickle load embeddings and adversarial embeddings

# Get task names
with open('datasets.json', 'r') as file:
    datasets = json.load(file)
tasks_name = []
for task in datasets:
    tasks_name.append(task["name"])

models_name = ["efficient_latent", "GURA.fusion_hubert_xlarge", "GURA.fusion_wav2vec2", "GURA.fusion_cat_xwc", "GURA.avg_xwc", "hear21passt.base", "hearbaseline.wav2vec2", "audio_dbert", "panns_hear"]
attack_types = ["Boundary", "HopSkipJump"]

clean_embeddings = {}
adversarial_embeddings = {}

# Load Dataset (Read dictionary pkl file)
for model_name in models_name:
    clean_embeddings[model_name] = {}
    adversarial_embeddings[model_name] = {}

    for task_name in tasks_name:
        clean_embeddings[model_name][task_name] = []
        adversarial_embeddings[model_name][task_name] = {}

        # Get metadata
        embeddings_path = f"embeddings/{model_name}/{task_name}"
        metadata_path = f"{embeddings_path}/task_metadata.json"

        with open(metadata_path, 'r') as file:
            metadata = json.load(file)

        if metadata["split_mode"] == "new_split_kfold" or metadata["split_mode"] == "presplit_kfold":
            split = "folds"
            n_folds = metadata["nfolds"]
        elif metadata["split_mode"] == "trainvaltest":
            split = "TVT"
            n_folds = None

        # Get clean embeddings
        embeddings = get_embeddings(embeddings_path, split, n_folds)
        clean_embeddings[model_name][task_name].append(embeddings)

        for attack_type in attack_types:
            adversarial_embeddings[model_name][task_name][attack_type] = []

            # Get adversarial_embeddings
            with open(f'adversarial_computations/{model_name}/{task_name}/{attack_type}_adv_embeddings.pkl', 'rb') as fp:
                print(f"\n- {attack_type} adv embeddings -")
                adv_embeddings = pickle.load(fp)
                adversarial_embeddings[model_name][task_name][attack_type].append(adv_embeddings)

            print("")


# In[4]:


# Get input data for SVM: clean embeddings and adversarial embeddings (Boundary and HopSkipJump)
x_clean_embeddings = {}
for model in clean_embeddings:
    x_clean_embeddings[model] = {}
    for task in clean_embeddings[model]:
        x_clean_embeddings[model][task] = []
        for split in clean_embeddings[model][task][0]:
            print(f"Get Clean X: {model} - {task} - {split}")
            x_clean_embeddings[model][task].append(clean_embeddings[model][task][0][split])

x_adversarial_embeddings = {}
for model in adversarial_embeddings:
    x_adversarial_embeddings[model] = {}
    for task in adversarial_embeddings[model]:
        x_adversarial_embeddings[model][task] = {}
        for attack in adversarial_embeddings[model][task]:
            x_adversarial_embeddings[model][task][attack] = []
            for split in adversarial_embeddings[model][task][attack][0]:
                print(f"Get Adv X: {model} - {task} - {attack} - {split}")
                x_adversarial_embeddings[model][task][attack].append(adversarial_embeddings[model][task][attack][0][split])


# In[5]:


# Input shape example
print(np.array(x_clean_embeddings["audio_dbert"]["esc50"], dtype='object').shape)
print(np.array(x_adversarial_embeddings["audio_dbert"]["esc50"]["Boundary"], dtype='object').shape)
print(np.array(x_adversarial_embeddings["audio_dbert"]["esc50"]["HopSkipJump"], dtype='object').shape)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

def SVM(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray
):
    """ Train an SVM on the train data portion and evaluate it on the test data portion.
    
    Parameters
    ------------------------
    X_train: np.ndarray
        Train inputs
    X_test: np.ndarray
        Test inputs
    Y_train: np.ndarray
        Train labels
    Y_test: np.ndarray
        Test labels
        
    Returns
    ------------------------
    accuracy: float
        The accuracy score over the test predictions
    f1_score: float
        The F1 score over the test predictions
    """
    
    # Initialize SVM classifier
    svm_classifier = SVC(kernel='rbf')  # rbf', 'poly', 'sigmoid'
    
    # Train the SVM classifier
    svm_classifier.fit(X_train, Y_train)
    
    # Predict for test data
    Y_pred = svm_classifier.predict(X_test)

    # Compute accuracy and F1 score metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    return accuracy, f1


# #### Main SVM training and evaluation loop
# Stores the results in *results_SVM*

# In[12]:


# Notes
# The SVM is trained and evaluated 1 time for each model-task-attack combination.
# In case of a TVT split the test fold is further split into train-test (only on this set the aversarial embeddings were computed) (still large enought for an accurate evaluation).
# In case of a K-fold split the test fold is fold00 and the rest are for training (no need for k-fold CV.
#  Given the data size and the fact that each fold is balanced with respect to the classes a single test is enought).

SVM_results = {}

for model in x_clean_embeddings:
    SVM_results[model] = {}
    for task in x_clean_embeddings[model]:
        SVM_results[model][task] = {}
        for attack in ["Boundary", "HopSkipJump"]:
            SVM_results[model][task][attack] = {}
            # Get train, test data split
            # TVT Split
            if len(x_clean_embeddings[model][task]) == 1:
                X_clean = np.array(x_clean_embeddings[model][task][0])
                X_adv = np.array(x_adversarial_embeddings[model][task][attack][0])

                X = np.concatenate((X_clean, X_adv), axis=0)
                Y = np.concatenate((np.zeros(X_clean.shape[0]), np.ones(X_adv.shape[0])), axis=0)

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            # K-fold split
            else:
                for n_fold in range(len(x_clean_embeddings[model][task])):
                    X_clean = np.array(x_clean_embeddings[model][task][n_fold])
                    X_adv = np.array(x_adversarial_embeddings[model][task][attack][n_fold])

                    # Test fold
                    if n_fold == 0:
                        X_test = np.concatenate((X_clean, X_adv), axis=0)
                        Y_test = np.concatenate((np.zeros(X_clean.shape[0]), np.ones(X_adv.shape[0])), axis=0)

                    # Train folds
                    elif n_fold == 1:
                        X_train = np.concatenate((X_clean, X_adv), axis=0)
                        Y_train = np.concatenate((np.zeros(X_clean.shape[0]), np.ones(X_adv.shape[0])), axis=0)
                    else:
                        X_fold = np.concatenate((X_clean, X_adv), axis=0)
                        Y_fold = np.concatenate((np.zeros(X_clean.shape[0]), np.ones(X_adv.shape[0])), axis=0)

                        X_train = np.concatenate((X_train, X_fold), axis=0)
                        Y_train = np.concatenate((Y_train, Y_fold), axis=0)

                # Shuffle the arrays
                train_indices = np.arange(len(X_train))
                test_indices = np.arange(len(X_test))

                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)

                X_train = X_train[train_indices]
                Y_train = Y_train[train_indices]
                X_test = X_test[test_indices]
                Y_test = Y_test[test_indices]

            # Train SVM and get results
            accuracy, f1 = SVM(X_train, X_test, Y_train, Y_test)

            print(f"-- Main SVM loop for {model} on task {task} over {attack} embeddings --\n-Results: Accuracy: {accuracy:.4f}, F1_Score: {f1:.4f}\n")

            SVM_results[model][task][attack]["Accuracy"] = []
            SVM_results[model][task][attack]["F1_Score"] = []

            SVM_results[model][task][attack]["Accuracy"].append(accuracy)
            SVM_results[model][task][attack]["F1_Score"].append(f1)


# In[13]:


SVM_results


# In[16]:


# Pickle save SVM_results
SVM_results_path = f"results_SVM"
if not(os.path.exists(SVM_results_path)):
    os.makedirs(SVM_results_path)
with open(f'{SVM_results_path}/SVM_results.pkl', 'wb') as fp:
    pickle.dump(SVM_results, fp)
    print(f"\n- (Save) SVM metrics stored in results")


# In[ ]:





# ### Results

# #### Per model

# In[4]:


# Pickle load SVM_results
with open(f'results_SVM/SVM_results.pkl', 'rb') as fp:
    SVM_results = pickle.load(fp)


# In[5]:


SVM_model_results = {}
for model in SVM_results:
    SVM_model_results[model] = {}
    for task in SVM_results[model]:
        for attack in SVM_results[model][task]:
            SVM_model_results[model][attack] = {}
            for metric in ["Accuracy", "F1_Score"]:
                SVM_model_results[model][attack][metric] = []

for model in SVM_results:
    for task in SVM_results[model]:
        for attack in SVM_results[model][task]:
            SVM_model_results[model][attack]["Accuracy"].append(SVM_results[model][task][attack]["Accuracy"])
            SVM_model_results[model][attack]["F1_Score"].append(SVM_results[model][task][attack]["F1_Score"])


# In[6]:


SVM_model_results


# In[7]:


models = []
accuracy_B = []
f1_score_B = []
accuracy_H = []
f1_score_H = []

# Extract data for barplot
for model, tasks in SVM_model_results.items():
    for attack, metrics in tasks.items():
        if attack == "Boundary":
            models.extend([model] * len(metrics['Accuracy']))
            accuracy_B.extend([item for sublist in metrics['Accuracy'] for item in sublist])
            f1_score_B.extend([item for sublist in metrics['F1_Score'] for item in sublist])
        elif attack == "HopSkipJump":
            accuracy_H.extend([item for sublist in metrics['Accuracy'] for item in sublist])
            f1_score_H.extend([item for sublist in metrics['F1_Score'] for item in sublist])

# Format as DataFrame
barplot_model_DF = pd.DataFrame({
    'Model': models,
    'Boundary Accuracy': accuracy_B,
    'Boundary F1 score': f1_score_B,
    'HopSkipJump Accuracy': accuracy_H,
    'HopSkipJump F1 score': f1_score_H
})

print(barplot_model_DF)


# In[13]:


from barplots import barplots

barplots(
    barplot_model_DF,
    groupby=["Model"],
    orientation="horizontal",
    height=12,
    legend_position="center left"
)
plt.savefig(f'{image_dir}/{script_name}_model_barplot_{timestamp}.png')


# ### Note on embeddings

# In[6]:


# In general the SVM has a hard time detecting the perturbation and dinstinghuishing clean vs adversarial embeddings
# Exception: Panns Hear model
# As shown here the Panns Hear model's embeddings have 0.0 as min and frequent value, this, which already made the model the most robust across all models,
# renders the identification of perturbation much easier.
# This further validates the fact that embeddings that are normalized/within a fixed min-max range handle better this kind of attack.

# Check of the actual clean and adversarial embeddings (first 100 values for GTZAN_music task) and prepare data for line plot
models_names = ["GURA.fusion_hubert_xlarge", "hear21passt.base", "panns_hear"]

clean_emb_plot = []
adv_emb_plot = []
for task in clean_embeddings["GURA.fusion_hubert_xlarge"]:
    if task == "GTZAN_music":
        for model_name in models_names:
            print(f"{model_name} -- {task} (clean and adversarial)")
            rounded_vector = [round(num, 4) for num in clean_embeddings[model_name][task][0]["fold00"][0][:100]]
            print(rounded_vector)
            clean_emb_plot.append(rounded_vector)
            print("-"*10)
            rounded_vector = [round(num, 4) for num in adversarial_embeddings[model_name][task]["Boundary"][0]["fold00"][0][:100]]
            print(rounded_vector)
            adv_emb_plot.append(rounded_vector)
            print("\n")

diff_embeddings = [np.array(adv) - np.array(clean) for adv, clean in zip(adv_emb_plot, clean_emb_plot)]


# In[13]:


import matplotlib.pyplot as plt
# Plotting line plot for differences across dimensions
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['blue', 'green', 'red']
for i, model in enumerate(models_names):
    x = np.arange(len(clean_emb_plot[i]))
    ax.plot(x, diff_embeddings[i], label=f'{model}', marker='o', color=colors[i])

ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('Differences between Clean and Adversarial Embedding values', fontsize=17)
ax.set_xticks(x)
ax.set_xticklabels("")
ax.set_ylabel('Difference Value', fontsize=15)
ax.set_xlabel('Embedding Index', fontsize=15)
ax.legend(fontsize=13)

plt.tick_params(bottom=False)
plt.tight_layout()
plt.savefig(f'{image_dir}/{script_name}_embLinePlot_{timestamp}.png', format='png')
plt.show()


# #### Per Task

# In[9]:


# Get task names
with open('datasets.json', 'r') as file:
    datasets = json.load(file)
tasks_name = []
for task in datasets:
    tasks_name.append(task["name"])

SVM_task_results = {}
for task in tasks_name:
    SVM_task_results[task] = {} 
    for model in SVM_results:
        for attack in SVM_results[model][task]:
            SVM_task_results[task][attack] = {}
            for metric in ["Accuracy", "F1_Score"]:
                SVM_task_results[task][attack][metric] = []

for task in tasks_name:
    for model in SVM_results:
        for attack in SVM_results[model][task]:
            SVM_task_results[task][attack]["Accuracy"].append(SVM_results[model][task][attack]["Accuracy"])
            SVM_task_results[task][attack]["F1_Score"].append(SVM_results[model][task][attack]["F1_Score"])


# In[10]:


SVM_task_results


# In[11]:


tasks = []
accuracy_B = []
f1_score_B = []
accuracy_H = []
f1_score_H = []

# Extract data for barplot
for task, models in SVM_task_results.items():
    for attack, metrics in models.items():
        if attack == "Boundary":
            tasks.extend([task] * len(metrics['Accuracy']))
            accuracy_B.extend([item for sublist in metrics['Accuracy'] for item in sublist])
            f1_score_B.extend([item for sublist in metrics['F1_Score'] for item in sublist])
        elif attack == "HopSkipJump":
            accuracy_H.extend([item for sublist in metrics['Accuracy'] for item in sublist])
            f1_score_H.extend([item for sublist in metrics['F1_Score'] for item in sublist])

# Format as DataFrame
barplot_task_DF = pd.DataFrame({
    'Task': tasks,
    'Boundary Accuracy': accuracy_B,
    'Boundary F1 score': f1_score_B,
    'HopSkipJump Accuracy': accuracy_H,
    'HopSkipJump F1 score': f1_score_H
})

print(barplot_task_DF)


# In[12]:


from barplots import barplots

barplots(
    barplot_task_DF,
    groupby=["Task"],
    orientation="horizontal",
    height=12,
    legend_position="center left"
)
plt.savefig(f'{image_dir}/{script_name}_task_barplot_{timestamp}.png')
plt.savefig(f'{image_dir}/{script_name}_task_barplot_{timestamp}.png')


# In[ ]:




