#!/bin/bash
set -e

# Evaluates the AAAI models against the test set from SciTail
# If you train a different model, change the archive_file flag to point to your model
# If you want to evaluate against a different test file (with similar vocabulary), change the
# evaluation_data_file flag.
# If you want to evaluate on a different dataset, train the model first using train_models.sh and
# use the learned model file here

echo "# -----------------------------------------------"
echo "# Evaluating DGEM model"
echo "# -----------------------------------------------"
python scitail/run.py evaluate \
  --archive_file SciTailModelsV1/dgem/model.tar.gz \
  --evaluation_data_file SciTailV1.1/dgem_format/scitail_1.0_structure_test.tsv

echo "\n\n# -----------------------------------------------"
echo "# Evaluating Decomposable Attention model"
echo "# -----------------------------------------------"
python scitail/run.py evaluate \
  --archive_file SciTailModelsV1/decompatt/model.tar.gz \
  --evaluation_data_file SciTailV1.1/snli_format/scitail_1.0_test.txt

echo "\n\n# -----------------------------------------------"
echo "# Evaluating Ngram Overlap model"
echo "# -----------------------------------------------"
python scitail/run.py evaluate \
  --archive_file SciTailModelsV1/simple_overlap/model.tar.gz \
  --evaluation_data_file SciTailV1.1/tsv_format/scitail_1.0_test.tsv

