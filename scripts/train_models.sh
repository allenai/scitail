#!/bin/bash
set -e

# Trains the DGEM and baseline models from AAAI on the SciTail dataset
# If you want to train on a different dataset, update the .json file corresponding to each model in
# training_config/ to point to the appropriate train/dev files. If you are using a GPU machine, set
# `cuda_device' to 0 in the .json files too.
# Use evaluate_models.sh to evaluate the learned models

# Folder where all the models will be saved
model_folder=models/

echo "*** Set cuda_device to 0 in the training_config/*.json files on a GPU for faster training ***"

echo "# -----------------------------------------------"
echo "# Training DGEM model"
echo "# -----------------------------------------------"
python scitail/run.py train \
  -s $model_folder/dgem/model.tar.gz \
  training_config/dgem.json

echo "\n\n# -----------------------------------------------"
echo "# Training Decomposable Attention model"
echo "# -----------------------------------------------"
python scitail/run.py train \
  -s $model_folder/decompatt/model.tar.gz \
  training_config/decompatt.json

echo "\n\n# -----------------------------------------------"
echo "# Training Ngram Overlap model"
echo "# -----------------------------------------------"
python scitail/run.py train \
  -s $model_folder/simple_overlap/model.tar.gz \
  training_config/simple_overlap.json

