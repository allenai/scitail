#!/bin/bash
set -e

DATASET_URL=http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip
MODELS_URL=https://s3-us-west-2.amazonaws.com/aristo-scitail/SciTailModelsV1.zip

echo "Downloading dataset"
wget $DATASET_URL

echo "Downloading models"
wget $MODELS_URL

echo "Decompressing zip files to `pwd`"
unzip -q SciTailV1.1.zip
unzip -q SciTailModelsV1.zip
