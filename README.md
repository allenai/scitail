# Scitail
SciTail (Science Entailment) dataset and models

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


6. Test installation

 ```
 pytest -v
 ```


# Download the data and models
Run the `download_data.sh` script to download the dataset and models used in the SciTail paper.
  ```
   sh scripts/download_data.sh
  ```

This will download and unzip the data to `SciTailV1` folder and models to `SciTailModelsV1` folder.


# Evaluate the SciTail models
To run the trained models on the test sets, run
  ```
    sh scripts/evaluate_models.sh
  ```

Note that the models include the vocabulary used for training these models. So these
pre-trained models will have poor performance on new test sets with a different vocabulary.

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