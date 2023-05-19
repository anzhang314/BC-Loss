# BC-Loss


## Overview

Official code of "Incorporating Bias-aware Margins into Contrastive Loss for Collaborative Filtering" (2022 NeurIPS)


## Run the Code

### Docker 
Run below commands to build image and start container
```
docker build -t bcloss .
docker run -dit -P -u $(id -u):$(id -g) -v /your/local/path/to/BC-Loss:/workspace/BC-Loss --gpus '"device=1"' --shm-size 16g --name bcloss_run bcloss:latest
```

- We provide implementation for various baselines presented in the paper.

- We also provide the In-Distribution(test_id) and Out-of-Distribution(test_ood) test splits for Amazon-book, Tencent and Alibaba-Ifashion datasets, and temporal split for Douban.

- To run the code, first run the following command to install tools used in evaluation:
```
python setup.py build_ext --inplace
```


### MF backbone
For models with MF as backbone, use models with random negative sampling strategy. For example:

- MFBPR Training(equivalent to 0 layer LightGCN):

```
python main.py --modeltype LGN --dataset tencent.new --n_layers 0 --neg_sample 1
```

- INFONCE Training:

```
python main.py --modeltype INFONCE --dataset tencent.new  --n_layers 0 --neg_sample 128
```

- BC-LOSS Training:

```
python main.py --modeltype BC_LOSS --dataset tencent.new --n_layers 0 --neg_sample 128
```


### LightGCN backbone
For models with LightGCN as backbone, use models with in-batch negative sampling strategy. For example:

- LightGCN Training:

```
python main.py --modeltype LGN --dataset tencent.new --n_layers 2 --neg_sample 1
```

- INFONCE Training:

```
python main.py --modeltype INFONCE_batch --dataset tencent.new  --n_layers 2 --neg_sample -1
```

- BC-LOSS Training:

```
python main.py --modeltype BC_LOSS_batch --dataset tencent.new --n_layers 2 --neg_sample -1
```

Details of hyperparameter settings for various baselines can be found in the paper.


## Requirements

- python == 3.7.10

- tensorflow == 1.14

- pytorch == 1.9.1+cu102


## Reference
If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{bc_loss,   
      author    = {An Zhang and
                   Wenchang Ma and 
                   Xiang Wang and 
                   Tat-seng Chua}, 
      title     = {Incorporating Bias-aware Margins into Contrastive Loss for Collaborative Filtering},  
      booktitle = {{NeurIPS}},  
      year      = {2022},   
}
```










