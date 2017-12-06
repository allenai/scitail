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


# Train the SciTail models
TBD

# Evaluate the SciTail models
TBD

If you find these models helpful in your work, please cite:
```
@inproceedings{scitail,
	  Author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
     Booktitle = {AAAI},
     Title = {SciTail: A Textual Entailment Dataset from Science Question Answering},
     Year = {2018}
}
```