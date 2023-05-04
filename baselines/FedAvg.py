import copy
import random
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Optional

import dgl
import torch as th
import torch.nn.functional as F
import tqdm

from baselines.Decoders import NodeClassifier
from baselines.RGCN import RGCN
from utils import get_data_dict, align_schemas, load_data, EarlyStopping, evaluate_node_classification


# FedAvg client
class Client:
    def __init__(self, args: Namespace, data: tuple, ntypes: list[str], etypes: list[str],
                 canonical_etypes: list[tuple[str, str, str]], client_id: int) -> None:
        self.ntypes = ntypes
        self.etypes = etypes
        self.canonical_etypes = canonical_etypes
        self.id = client_id
        self.lr = args.lr
        self.optim = args.optim
        self.weight_decay = args.weight_decay
        self.num_local_epochs = args.num_local_epochs
        self.mu = args.mu
        self.task = args.task
        self.device = args.device

        # Use GPU-based neighborhood sampling if possible
        num_workers = 4 if args.device.type == "cpu" else 0
        if self.task == "node_classification":
            g, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = data
            self.g = g.to(self.device)
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
                batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=num_workers, device=args.device,
                use_uva=False)
            self.val_dataloader = dgl.dataloading.DataLoader(self.g, self.val_nid_dict, sampler,
                batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=num_workers, device=args.device,
                use_uva=False)
            self.test_dataloader = dgl.dataloading.DataLoader(self.g, self.test_nid_dict, sampler,
                batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=num_workers, device=args.device,
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

    def local_update(self) -> float:
        # deep copy the local encoder and decoder models
        global_encoder = copy.deepcopy(self.encoder)
        global_decoder = copy.deepcopy(self.decoder)

        if self.optim == "Adam":
            optimizer = th.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
                                      weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            optimizer = th.optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr,
                                     weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer: {}".format(self.optim))

        self.encoder.train()
        self.decoder.train()
        avg_epoch_loss = 0
        with tqdm.tqdm(range(self.num_local_epochs), desc=f"Client {self.id}") as tq:
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
                        proximal_term = self.compute_proximal_term(global_encoder, global_decoder) * self.mu / 2

                        optimizer.zero_grad()
                        (batch_loss + proximal_term).backward()
                        optimizer.step()

                        epoch_loss += batch_loss.item() * logits.shape[0]
                        num_samples += logits.shape[0]
                    epoch_loss /= num_samples
                    # print training info
                    tq.set_postfix({"epoch-loss": f"{epoch_loss:.4f}"}, refresh=False)
                    avg_epoch_loss += epoch_loss
            else:
                raise ValueError("Unknown task: {}".format(self.task))

        avg_epoch_loss /= self.num_local_epochs
        return avg_epoch_loss

    def local_evaluate(self, is_test: bool = False) -> dict[str, float]:
        if self.task == "node_classification":
            dataloader = self.test_dataloader if is_test else self.val_dataloader
            results = evaluate_node_classification(self.encoder, self.decoder, dataloader, self.target_ntype)
        else:
            raise ValueError("Unknown task: {}".format(self.task))
        return results

    def compute_proximal_term(self, global_encoder, global_decoder):
        # compute proximal term
        proximal_term = 0
        for param_name, param in self.encoder.named_parameters():
            if not param_name.startswith("embed_layer"):
                proximal_term += th.sum((param - global_encoder.get_parameter(param_name).detach()) ** 2)
        for param, global_param in zip(self.decoder.parameters(), global_decoder.parameters()):
            proximal_term += th.sum((param - global_param.detach()) ** 2)
        return proximal_term


# FedAvg server
class Server:
    def __init__(self, args: Namespace, ntypes: list[str], etypes: list[str],
                 canonical_etypes: list[tuple[str, str, str]], out_dim: Optional[int] = None) -> None:
        self.num_clients = args.num_clients
        self.ntypes = ntypes
        self.etypes = etypes
        self.canonical_etypes = canonical_etypes
        # obtain initial state_dict from a dummy HGNN model
        if args.model == "RGCN":
            dummy_encoder = RGCN(args.hidden_dim, args.hidden_dim, self.etypes, {"ntype": 1}, args.num_bases,
                                 num_hidden_layers=args.num_layers - 2, dropout=args.dropout,
                                 use_self_loop=args.use_self_loop)
        else:
            raise ValueError("Unknown model: {}".format(args.model))
        dummy_encoder.to(args.device)
        state_dict_encoder = dummy_encoder.state_dict()
        # discard the embedding layer
        keys_to_remove = []
        for key in state_dict_encoder:
            if key.startswith("embed_layer"):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del state_dict_encoder[key]
        self.state_dict_encoder = state_dict_encoder

        # obtain initial state_dict from a dummy decoder model
        if args.task == "node_classification":
            assert isinstance(out_dim, int)
            dummy_decoder = NodeClassifier(args.hidden_dim, out_dim)
        else:
            raise ValueError("Unknown task: {}".format(args.task))
        dummy_decoder.to(args.device)
        self.state_dict_decoder = dummy_decoder.state_dict()

    def send_model(self, client: Client) -> None:
        # send state_dict
        client.encoder.load_state_dict(self.state_dict_encoder, strict=False)
        client.decoder.load_state_dict(self.state_dict_decoder, strict=False)

    def aggregate_model(self, clients: list[Client], client_weights: Optional[list[float]] = None) -> None:
        # aggregate state_dict_encoder and state_dict_decoder
        state_dict_encoder_list = [client.encoder.state_dict() for client in clients]
        state_dict_decoder_list = [client.decoder.state_dict() for client in clients]
        client_weights = [1.0 / len(clients) for _ in range(len(clients))] if client_weights is None else client_weights
        for key in self.state_dict_encoder:
            self.state_dict_encoder[key] = th.sum(
                th.stack([client_weights[i] * state_dict_encoder_list[i][key] for i in range(len(clients))]), dim=0)
        for key in self.state_dict_decoder:
            self.state_dict_decoder[key] = th.sum(
                th.stack([client_weights[i] * state_dict_decoder_list[i][key] for i in range(len(clients))]), dim=0)


class FedAvg:
    def __init__(self, args: Namespace) -> None:
        self.max_rounds = args.max_rounds
        self.num_clients = args.num_clients
        self.fraction = args.fraction
        self.task = args.task
        self.val_interval = args.val_interval
        self.patience = args.patience
        self.save_path = args.save_path

        if self.task == "node_classification":
            # data preparation for all clients
            g_list, out_dim, train_nid_dict_list, val_nid_dict_list, test_nid_dict_list = load_data(args)
            # align schemas
            ntypes, etypes, canonical_etypes = align_schemas(g_list)
            # setup clients
            self.clients = [
                Client(args, (g_list[i], out_dim, train_nid_dict_list[i], val_nid_dict_list[i], test_nid_dict_list[i]),
                       ntypes, etypes, canonical_etypes, i) for i in range(self.num_clients)]
            # setup a server
            self.server = Server(args, ntypes, etypes, canonical_etypes, out_dim)
            # client weights for updates and evaluation metrics
            self.train_client_weights = [sum([len(nids) for nids in nid_dict.values()]) for nid_dict in
                                         train_nid_dict_list]
            train_num_samples = sum(self.train_client_weights)
            self.train_client_weights = [weight / train_num_samples for weight in self.train_client_weights]
            self.val_client_weights = [sum([len(nids) for nids in nid_dict.values()]) for nid_dict in val_nid_dict_list]
            val_num_samples = sum(self.val_client_weights)
            self.val_client_weights = [weight / val_num_samples for weight in self.val_client_weights]
            self.test_client_weights = [sum([len(nids) for nids in nid_dict.values()]) for nid_dict in
                                        test_nid_dict_list]
            test_num_samples = sum(self.test_client_weights)
            self.test_client_weights = [weight / test_num_samples for weight in self.test_client_weights]
        else:
            raise ValueError("Unknown task: {}".format(self.task))

    def train(self) -> None:
        sample_size = max(round(self.fraction * self.num_clients), 1)
        early_stopping = EarlyStopping(patience=self.patience, mode="score", save_path=self.save_path, verbose=True)
        with tqdm.tqdm(range(self.max_rounds), desc="FedAvg") as tq:
            for round_no in tq:
                # randomly select clients
                selected_clients = random.sample(self.clients, sample_size)
                # server sends model to clients
                for client in selected_clients:
                    self.server.send_model(client)
                # clients train on their local data (local update)
                round_loss = 0
                for client in selected_clients:
                    client_loss = client.local_update()
                    round_loss += client_loss
                round_loss /= sample_size
                # server aggregates model updates from clients
                selected_clients_weights = [self.train_client_weights[client.id] for client in selected_clients]
                total_weight = sum(selected_clients_weights)
                selected_clients_weights = [weight / total_weight for weight in selected_clients_weights]
                self.server.aggregate_model(selected_clients, selected_clients_weights)

                tq.set_postfix({"round-loss": f"{round_loss:.4f}"}, refresh=False)

                # validation and early stopping
                if (round_no + 1) % self.val_interval == 0:
                    val_results = self.evaluate(is_test=False)
                    print_info = {key: f"{value:.4f}" for key, value in val_results.items()}
                    tq.set_postfix(print_info, refresh=False)
                    if self.task == "node_classification":
                        # quantity = (val_results["macro-f1"] + val_results["micro-f1"]) / 2
                        quantity = val_results["accuracy"]
                    else:
                        raise ValueError("Unknown task: {}".format(self.task))
                    early_stopping(quantity, callback=self.save_checkpoint)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

    def evaluate(self, is_test: bool = False) -> dict[str, float]:
        # send the latest model to all clients
        for client in self.clients:
            self.server.send_model(client)
        # get client weights
        client_weights = self.test_client_weights if is_test else self.val_client_weights
        # clients evaluate on their local data
        avg_results = defaultdict(float)
        for client, weight in zip(self.clients, client_weights):
            client_results = client.local_evaluate(is_test)
            for k, v in client_results.items():
                avg_results[k] += v * weight
        return dict(avg_results)

    def save_checkpoint(self, save_path: str) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        # save server model
        th.save(self.server.state_dict_encoder, save_path / "server_encoder.pt")
        th.save(self.server.state_dict_decoder, save_path / "server_decoder.pt")
        # save clients model
        for i, client in enumerate(self.clients):
            th.save(client.encoder.state_dict(), save_path / f"client_{i}_encoder.pt")
            th.save(client.decoder.state_dict(), save_path / f"client_{i}_decoder.pt")

    def load_checkpoint(self, load_path: str) -> None:
        load_path = Path(load_path)
        # load server model
        self.server.state_dict_encoder = th.load(load_path / "server_encoder.pt")
        self.server.state_dict_decoder = th.load(load_path / "server_decoder.pt")
        # load clients model
        for i, client in enumerate(self.clients):
            client.encoder.load_state_dict(th.load(load_path / f"client_{i}_encoder.pt"))
            client.decoder.load_state_dict(th.load(load_path / f"client_{i}_decoder.pt"))
