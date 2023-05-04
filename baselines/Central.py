from pathlib import Path
from typing import Optional

import dgl
import torch as th
import torch.nn.functional as F
import tqdm

from baselines.Decoders import NodeClassifier
from baselines.RGCN import RGCN
from utils import get_data_dict, load_data, EarlyStopping, evaluate_node_classification


class Central:
    def __init__(self, args, name: str = "Central", data: Optional[tuple] = None):
        self.name = name
        self.lr = args.lr
        self.optim = args.optim
        self.weight_decay = args.weight_decay
        self.max_epochs = args.max_epochs
        self.val_interval = args.val_interval
        self.patience = args.patience
        self.save_path = args.save_path
        self.task = args.task
        self.device = args.device

        # Use GPU-based neighborhood sampling if possible
        num_workers = 4 if args.device.type == "cpu" else 0
        if self.task == "node_classification":
            if data is None:
                [g], out_dim, [train_nid_dict], [val_nid_dict], [test_nid_dict] = load_data(args)
            else:
                g, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = data
            self.g = g.to(self.device)
            self.ntypes = g.ntypes
            self.etypes = list(dict.fromkeys(g.etypes))
            self.canonical_etypes = g.canonical_etypes
            self.out_dim = out_dim
            self.train_nid_dict = {k: v.to(self.device) for k, v in train_nid_dict.items()}
            self.val_nid_dict = {k: v.to(self.device) for k, v in val_nid_dict.items()}
            self.test_nid_dict = {k: v.to(self.device) for k, v in test_nid_dict.items()}
            self.num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
            assert len(self.g.ntypes) == 1 or len(self.g.ndata["y"].keys()) == 1
            assert len(self.train_nid_dict.keys()) == 1
            assert len(self.val_nid_dict.keys()) == 1
            assert len(self.test_nid_dict.keys()) == 1

            self.target_ntype = list(self.train_nid_dict.keys())[0]

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
            self.train_dataloader = dgl.dataloading.DataLoader(self.g, self.train_nid_dict, sampler,
                                                               batch_size=args.batch_size, shuffle=True,
                                                               drop_last=False, num_workers=num_workers,
                                                               device=args.device,
                                                               use_uva=False)
            self.val_dataloader = dgl.dataloading.DataLoader(self.g, self.val_nid_dict, sampler,
                                                             batch_size=args.batch_size, shuffle=False, drop_last=False,
                                                             num_workers=num_workers, device=args.device,
                                                             use_uva=False)
            self.test_dataloader = dgl.dataloading.DataLoader(self.g, self.test_nid_dict, sampler,
                                                              batch_size=args.batch_size, shuffle=False,
                                                              drop_last=False, num_workers=num_workers,
                                                              device=args.device,
                                                              use_uva=False)

            if args.model == "RGCN":
                self.encoder = RGCN(args.hidden_dim, args.hidden_dim, self.etypes, self.num_nodes_dict, args.num_bases,
                                    num_hidden_layers=args.num_layers - 2, dropout=args.dropout,
                                    use_self_loop=args.use_self_loop)
            else:
                raise ValueError("Unknown model: {}".format(args.model))
            self.encoder.to(args.device)
            self.decoder = NodeClassifier(args.hidden_dim, self.out_dim)
            self.decoder.to(args.device)
        else:
            raise ValueError("Unknown task: {}".format(self.task))

    def train(self):
        if self.optim == "Adam":
            optimizer = th.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
                                      weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            optimizer = th.optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
                                     weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer: {}".format(self.optim))
        early_stopping = EarlyStopping(patience=self.patience, mode="score", save_path=self.save_path, verbose=True)

        self.encoder.train()
        self.decoder.train()
        with tqdm.tqdm(range(self.max_epochs), desc=self.name) as tq:
            if self.task == "node_classification":
                for epoch in tq:
                    epoch_loss = 0
                    num_samples = 0
                    for iteration, (input_nodes, output_nodes, blocks) in enumerate(self.train_dataloader):
                        input_features = get_data_dict(blocks[0].srcdata["x"], blocks[0].srctypes)
                        output_labels = get_data_dict(blocks[-1].dstdata["y"], blocks[-1].dsttypes)

                        h_dict = self.encoder(blocks, input_features)
                        logits = self.decoder(h_dict[self.target_ntype])
                        logp = F.log_softmax(logits, dim=-1)
                        batch_loss = F.nll_loss(logp, output_labels[self.target_ntype])

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()

                        epoch_loss += batch_loss.item() * logits.shape[0]
                        num_samples += logits.shape[0]
                    epoch_loss /= num_samples
                    # print training info
                    tq.set_postfix({"train-loss": f"{epoch_loss:.4f}"}, refresh=False)
                    # validation and early stopping
                    if (epoch + 1) % self.val_interval == 0:
                        val_results = self.evaluate(is_test=False)
                        print_info = {key: f"{value:.4f}" for key, value in val_results.items()}
                        tq.set_postfix(print_info, refresh=False)
                        # quantity = (val_results["macro-f1"] + val_results["micro-f1"]) / 2
                        quantity = val_results["accuracy"]
                        early_stopping(quantity, callback=self.save_checkpoint)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break
            else:
                raise ValueError("Unknown task: {}".format(self.task))

    def evaluate(self, is_test=False):
        if self.task == "node_classification":
            dataloader = self.test_dataloader if is_test else self.val_dataloader
            results = evaluate_node_classification(self.encoder, self.decoder, dataloader, self.target_ntype)
        else:
            raise ValueError("Unknown task: {}".format(self.task))
        return results

    def save_checkpoint(self, save_path: str) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        th.save(self.encoder.state_dict(), save_path / f"{self.name}_encoder.pt")
        th.save(self.decoder.state_dict(), save_path / f"{self.name}_decoder.pt")

    def load_checkpoint(self, load_path: str) -> None:
        load_path = Path(load_path)
        self.encoder.load_state_dict(th.load(load_path / f"{self.name}_encoder.pt"))
        self.decoder.load_state_dict(th.load(load_path / f"{self.name}_decoder.pt"))
