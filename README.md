## Code for KOGCN

### 1. New Folders
create 2 new folders `glove`, `data_embedding`

### 2. Download Glove 
Download and put glove embeddings `glove.840B.300d.txt` in the new folder `glove`. This is only used as a placeholder, we do not use the glove embeddings in this project. 

### 3. Install Libraries
Install all libraries needed using `conda env create -f env.yaml`

### 4. Run Mams 

Use the default settings for `KOGCN`
```
python train.py --dataset mams --cs_dropout 0.2 --linear_dropout 0.3 --dep_dim 40 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 512
```

To use other datasets, you can change the dataset name `mams` to the desired one such as `rest14`, `laptop14`, `twitter`, `small`. 

### 5. Run Baselines 
Run `dot-GCN` 
```
python train.py --dataset mams
```
