## FedHGN

This repository provides a reference implementation of FedHGN as described in the following paper.
> FedHGN: A Federated Framework for Heterogeneous Graph Neural Networks.<br>
> Xinyu Fu, Irwin King

Accepted by IJCAI 2023. Will be available at ArXiv.

### Dependencies

* Python 3.9
* PyTorch 1.13.1
* DGL 0.9.1
* scikit-learn 1.1.2
* SciPy 1.9.0
* PyYAML 6.0
* tqdm 4.64.1

### Datasets

The raw data are obtained from DGL:
* [AIFB](https://docs.dgl.ai/en/0.9.x/generated/dgl.data.AIFBDataset.html)
* [MUTAG](https://docs.dgl.ai/en/0.9.x/generated/dgl.data.MUTAGDataset.html)
* [BGS](https://docs.dgl.ai/en/0.9.x/generated/dgl.data.BGSDataset.html)

The datasets above are preprocessed by [prepare_data.ipynb](prepare_data.ipynb).

### Usage

```
usage: main.py [-h] --dataset DATASET [--split-strategy SPLIT_STRATEGY] [--framework FRAMEWORK] [--ablation ABLATION]
               [--model MODEL] [--num-clients NUM_CLIENTS] [--gpu GPU] [--random-seed RANDOM_SEED]
               [--config-path CONFIG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        name of dataset
  --split-strategy SPLIT_STRATEGY, -s SPLIT_STRATEGY
                        dataset splitting strategy, either "edges" or "etypes"
  --framework FRAMEWORK, -f FRAMEWORK
                        FedHGN/FedAvg/FedProx/Local/Central
  --ablation ABLATION, -a ABLATION
                        B/C/B+C
  --num-clients NUM_CLIENTS, -c NUM_CLIENTS
                        number of clients, 3/5/10
  --gpu GPU, -g GPU     which gpu to use, specify -1 to use CPU
  --random-seed RANDOM_SEED
                        random seed
  --config-path CONFIG_PATH
                        path to config file
```

For example, to run FedHGN on the AIFB dataset with the random edges splitting strategy with 5 clients using GPU:
```
python main.py -d AIFB -s edges -s edges -f FedHGN -c 5 -g 0
```

### Citing

If you find FedHGN useful in your research, please cite the following paper:
```
To be available
```
