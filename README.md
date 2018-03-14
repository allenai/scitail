# Scitail
A repository of the the entailment models used for evaluation in the __SciTail: A Textual
Entailment Dataset from Science Question Answering__ paper accepted to AAAI'18. It contains
three models built using the PyTorch-based deep-learning NLP library, [AllenNLP](http://allennlp.org/).

 * Decomposable Attention (Baseline): A simple model that decomposes the
 problem into parallelizable attention computations ([Parikh et al. 2016](https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27)).
 We directly use the AllenNLP implementation ([Gardner, et al., 2017](http://allennlp.org/papers/AllenNLP_white_paper.pdf))
 of the decomposable attention model here.

 * Ngram Overlap (Baseline): A simple word-overlap baseline  that uses the proportion of unigrams,
 1- skip bigrams, and 1-skip trigrams in the hypothesis that are also present in the premise as
 three features. We feed these features into a two-layer perceptron.

 * __Decomposable Graph Entailment Model__ (Proposed): Our proposed decomposed graph entailment
 model that uses structure from the hypothesis to calculate entailment probabilities for each node
 and edge in the graph structure and aggregates them for the final entailment computation. Please
 refer to [our paper](http://ai2-website.s3.amazonaws.com/publications/scitail-aaai-2018_cameraready.pdf)
 for more details.

We use the [SciTail dataset](http://data.allenai.org/scitail/) and pre-trained models by default
(downloaded automatically by the `scripts/download_data.sh` script). The models can be trained and
evaluated on new datasets too as described below.


# Setup Instruction

1. Create the `scitail` environment using Anaconda

  ```
  conda create -n scitail python=3.6
  ```

2. Activate the environment

  ```
  source activate scitail
  ```

3. Install the requirements in the environment: 

  ```
  sh scripts/install_requirements.sh
  ```

4. Install pytorch as per instructions on <http://pytorch.org/>. Commands as of Nov. 22, 2017:

  Linux/Mac (no CUDA): `conda install pytorch torchvision -c soumith`

  Linux   (with CUDA): `conda install pytorch torchvision cuda80 -c soumith`


5. Download the [Glove embeddings](https://nlp.stanford.edu/projects/glove/) into a `Glove/`
folder in the root directory as `glove.<tokens>B.<dim>d.txt.gz` files.

6. Test installation

 ```
 pytest -v
 ```


# Download the data and models
Run the `download_data.sh` script to download the dataset and models used in the SciTail paper.
  ```
   sh scripts/download_data.sh
  ```

This will download and unzip the data to `SciTailV1.1` folder (from
 [here](http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip))
 and models to `SciTailModelsV1` folder (from
 [here](https://s3-us-west-2.amazonaws.com/aristo-scitail/SciTailModelsV1.zip)).


# Evaluate the SciTail models
To run the trained models on the test sets, run
  ```
    sh scripts/evaluate_models.sh
  ```

Note that the models include the vocabulary used for training these models. So these
pre-trained models will have poor performance on new test sets with a different vocabulary.

# View predictions of the SciTail models
To view the model predictions, run
  ```
    sh scripts/predict_model.sh
  ```
The predictions would be added to the `predictions/` folder for each model. Each file has the 
original examples along with the model's probability, logit predictions, and entailment score using 
the keys: `label_probs`, `label_logits` and `score` respectively.

# Train the SciTail models
To train the models on new datasets, run
   ```
     sh scripts/train_models.sh
   ```
with the appropriate train/validation sets specified in the training configuration files.
  * Decomposable Graph Entailment Model: `training_config/dgem.json`
  * Decomposable Attention Model: `training_config/decompatt.json`
  * NGram Overlap Model: `training_config/simple_overlap.json`


If you find these models helpful in your work, please cite:
```
@inproceedings{scitail,
     Author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
     Booktitle = {AAAI},
     Title = {{SciTail}: A Textual Entailment Dataset from Science Question Answering},
     Year = {2018}
}
```