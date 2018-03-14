#!/bin/bash
set -e

# Generates the predictions from the AAAI models against the $split set from SciTail
# If you train a different model, change the path to model.tar.gz to point to your model
# If you want predictions against a different test file (with similar vocabulary), change the
# jsonl file path (expected fields: gold_label, sentence1, sentence2, sentence2_structure [for
# dgem]). Please refer to the files in SciTailV1.1/predictor_format for examples.
# If you want to evaluate on a different dataset, train the model first using train_models.sh and
# use the learned model file here

split=dev

mkdir -p predictions

echo "# -----------------------------------------------"
echo "# Predicting DGEM model"
echo "# -----------------------------------------------"
python scitail/run.py predict \
  --silent --output-file predictions/scitail_1.0_dgem_predictions_$split.jsonl \
  SciTailModelsV1/dgem/model.tar.gz \
  SciTailV1.1/predictor_format/scitail_1.0_structure_$split.jsonl

echo "\n\n# -----------------------------------------------"
echo "# Predicting Decomposable Attention model"
echo "# -----------------------------------------------"
python scitail/run.py predict \
  --silent --output-file predictions/scitail_1.0_decompatt_predictions_$split.jsonl \
  SciTailModelsV1/decompatt/model.tar.gz \
  SciTailV1.1/predictor_format/scitail_1.0_structure_$split.jsonl


echo "\n\n# -----------------------------------------------"
echo "# Predicting Ngram Overlap model"
echo "# -----------------------------------------------"
python scitail/run.py predict \
  --silent --output-file predictions/scitail_1.0_overlap_predictions_$split.jsonl \
  SciTailModelsV1/simple_overlap/model.tar.gz \
  SciTailV1.1/predictor_format/scitail_1.0_structure_$split.jsonl


