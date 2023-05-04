import random
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import yaml
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score


def load_configs(args):
    with open(args.config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dataset_configs = configs["datasets"][args.dataset]
    dataset_cname = dataset_configs.pop("cname")
    dataset_path = dataset_configs.pop("path")
    if dataset_configs["task"] == "node_classification":
        task_literal = "nc"
    else:
        raise ValueError("Unknown task type: {}".format(dataset_configs["task"]))
    model_configs = configs["models"].get(f"{args.model}_{task_literal}", configs["models"][args.model])
    framework_configs = configs["frameworks"].get(f"{args.framework}_{task_literal}",
                                                  configs["frameworks"][args.framework])

    if args.framework == "Central":
        args.split_strategy = "centralized"
        args.num_clients = 1
    if args.framework != "FedHGN":
        args.ablation = None
    assert args.split_strategy in ["centralized", "edges", "etypes"]
    if args.split_strategy in ["edges", "etypes"]:
        args.split_strategy = f"random-{args.split_strategy}"

    all_configs = vars(args) | dataset_configs | model_configs | framework_configs
    all_configs["dataset_cname"] = dataset_cname
    all_configs["dataset_path"] = dataset_path

    return Namespace(**all_configs)


def get_save_path(args, prefix="./saves"):
    save_path = Path(prefix, args.framework if args.ablation is None else f"{args.framework}_{args.ablation}",
                     args.model, f"{args.dataset}_{args.split_strategy}_{args.num_clients}")
    save_path.mkdir(parents=True, exist_ok=True)
    old_saves = [int(str(x.name)) for x in save_path.iterdir() if x.is_dir() and str(x.name).isdigit()]
    if len(old_saves) == 0:
        save_num = 1
    else:
        save_num = max(old_saves) + 1
    save_path = save_path / str(save_num)
    save_path.mkdir()

    # copy config files to the save dir
    shutil.copy("./configs.yaml", save_path)

    return str(save_path)


def set_random_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def get_data_dict(data, types):
    if len(types) == 1:
        assert not isinstance(data, dict)
        return {types[0]: data}
    else:
        assert isinstance(data, dict)
        return data


def align_schemas(g_list):
    ntypes = []
    etypes = []
    canonical_etypes = []
    for g in g_list:
        ntypes.extend(g.ntypes)
        etypes.extend(g.etypes)
        canonical_etypes.extend(g.canonical_etypes)
    ntypes = list(set(ntypes))
    etypes = list(set(etypes))
    canonical_etypes = list(set(canonical_etypes))
    return ntypes, etypes, canonical_etypes


def print_results(results: dict[str, float]):
    print("\t".join(results.keys()))
    print("\t".join([f"{v:.4f}" for v in results.values()]))


def save_results(results: dict[str, float], save_path: str):
    save_path = Path(save_path)
    with save_path.joinpath("results.txt").open("w") as f:
        f.write("\t".join(results.keys()) + "\n")
        f.write("\t".join([f"{v:.4f}" for v in results.values()]) + "\n")


def evaluate_node_classification(encoder, decoder, dataloader, target_ntype):
    results = {}

    logits_list = []
    y_true_list = []
    encoder.eval()
    decoder.eval()
    with th.no_grad():
        for iteration, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            input_features = get_data_dict(blocks[0].srcdata["x"], blocks[0].srctypes)
            output_labels = get_data_dict(blocks[-1].dstdata["y"], blocks[-1].dsttypes)

            h_dict = encoder(blocks, input_features)
            logits = decoder(h_dict[target_ntype])

            logits_list.append(logits.cpu().numpy())
            y_true_list.append(output_labels[target_ntype].cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    y_pred = np.argmax(logits, axis=-1)
    y_score = softmax(logits, axis=-1)
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["macro-f1"] = f1_score(y_true, y_pred, average="macro")
    results["micro-f1"] = f1_score(y_true, y_pred, average="micro")
    # results["roc-auc"] = roc_auc_score(y_true, y_score, multi_class="ovr")

    return results


def load_data(args):
    if args.task == "node_classification":
        if args.dataset in ["AIFB", "MUTAG", "BGS"]:
            load_path = Path(args.dataset_path, f"{args.dataset_cname}_{args.split_strategy}_{args.num_clients}.bin")
            g_list, label_dict = dgl.load_graphs(str(load_path))

            g_list = [g.long() for g in g_list]
            for g in g_list:
                g.ndata["y"] = g.ndata["label"]
                g.ndata["x"] = g.ndata[dgl.NID]  # dummy node feature, will not be used

            out_dim = label_dict["num_classes"][0].item()

            train_nid_dict_list = [
                {ntype: train_mask.nonzero().flatten() for ntype, train_mask in g.ndata["train_mask"].items()} for g in
                g_list]
            val_nid_dict_list = [
                {ntype: val_mask.nonzero().flatten() for ntype, val_mask in g.ndata["val_mask"].items()} for g in
                g_list]
            test_nid_dict_list = [
                {ntype: test_mask.nonzero().flatten() for ntype, test_mask in g.ndata["test_mask"].items()} for g in
                g_list]
        else:
            raise ValueError("Unknown dataset of task {}: {}".format(args.task, args.dataset))
        return g_list, out_dim, train_nid_dict_list, val_nid_dict_list, test_nid_dict_list
    else:
        raise ValueError("Unknown task: {}".format(args.task))


class EarlyStopping:
    """Early stops the training if validation score/loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=1e-5, mode="score", save_path="checkpoint.pt", verbose=False, ):
        """
        Args:
            patience (int): How long to wait after last time validation score/loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation score/loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.Inf
        self.early_stop = False

    def __call__(self, quantity: float, model: Optional[nn.Module] = None, callback: Optional[Callable] = None):
        if self.mode == "score":
            score = quantity
        elif self.mode == "loss":
            score = -quantity
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(quantity, model, callback)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, quantity: float, model: Optional[nn.Module] = None, callback: Optional[Callable] = None):
        """Saves model when validation score/loss improves."""
        if self.verbose:
            if self.mode == "score":
                print(f"Validation score increased ({self.best_score:.6f} --> {quantity:.6f}).  Saving model ...")
            elif self.mode == "loss":
                print(f"Validation loss decreased ({-self.best_score:.6f} --> {quantity:.6f}).  Saving model ...")
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

        if model is not None:
            th.save(model.state_dict(), self.save_path)
        if callback is not None:
            callback(self.save_path)
