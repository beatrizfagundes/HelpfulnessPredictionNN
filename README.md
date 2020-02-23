# A Multitask Learning Network to Predict Helpfulness and Specificity of Reviews

## Dependencies
- Python 3.7
- Tensorflow 2.1.0
- Numpy 1.18.1
- Pandas 0.25.3
- Scikit-learn 0.22.1

To install the dependencies, you can simple run:
pip install -r requirements.txt OR pip3 install -r requirements.txt

Alternatively, if you have installed Anaconda, you can create an environment and install the dependencies inside of this environment only:
conda create -n newenv python=3.7
conda activate newenv
pip install -r requirements.txt

## How to run
### Single task neural network
python3 single_task_nn.py

## Multitask neural network
python3 multi_task.py
