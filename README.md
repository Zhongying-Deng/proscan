## Install environment

```
conda create -n py37 python=3.7.13
conda activate py37
pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

## Training

Provide the path to the training for the `data_path` variable in `train.sh`, and then run the following command
```bash
bash train.sh
```

Currently, the dataset has not been split into train/val/test sets. Only the training set is used (see Line 164 of `train_cross_val_prostate.py`).
