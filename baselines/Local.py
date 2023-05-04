from collections import defaultdict

from baselines.Central import Central
from utils import load_data


class Local:
    def __init__(self, args):
        self.num_clients = args.num_clients
        self.task = args.task
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.max_epochs = args.max_epochs
        self.val_interval = args.val_interval
        self.patience = args.patience
        self.save_path = args.save_path
        self.device = args.device

        if self.task == "node_classification":
            # data preparation for all clients
            g_list, out_dim, train_nid_dict_list, val_nid_dict_list, test_nid_dict_list = load_data(args)
            # setup clients
            self.clients = [Central(args, name=f"Local-{i}", data=(
                g_list[i], out_dim, train_nid_dict_list[i], val_nid_dict_list[i], test_nid_dict_list[i])) for i in
                            range(self.num_clients)]
            # client weights for testing
            self.test_client_weights = [sum([len(nids) for nids in nid_dict.values()]) for nid_dict in
                                        test_nid_dict_list]
            test_num_samples = sum(self.test_client_weights)
            self.test_client_weights = [weight / test_num_samples for weight in self.test_client_weights]
        else:
            raise ValueError("Unknown task: {}".format(self.task))

    def train(self):
        for client in self.clients:
            client.train()

    def evaluate(self, is_test: bool = False):
        results = defaultdict(float)
        for client, weight in zip(self.clients, self.test_client_weights):
            for key, value in client.evaluate(is_test).items():
                results[key] += value * weight
        return dict(results)

    def save_checkpoint(self, save_path: str) -> None:
        for client in self.clients:
            client.save_checkpoint(save_path)

    def load_checkpoint(self, load_path: str) -> None:
        for client in self.clients:
            client.load_checkpoint(load_path)
