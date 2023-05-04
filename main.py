import argparse

import torch as th

from FedHGN import FedHGN
from baselines.Central import Central
from baselines.FedAvg import FedAvg
from baselines.Local import Local
from utils import load_configs, get_save_path, set_random_seeds, print_results, save_results


def main(args):
    if args.framework == "FedHGN":
        fl_framework = FedHGN(args)
    elif args.framework in ["FedAvg", "FedProx"]:
        # FedAvg: mu = 0; FedProx: mu = 1
        fl_framework = FedAvg(args)
    elif args.framework == "Local":
        fl_framework = Local(args)
    elif args.framework == "Central":
        fl_framework = Central(args)
    else:
        raise ValueError("Unknown framework.")

    fl_framework.train()
    fl_framework.load_checkpoint(args.save_path)  # load the saved best model
    avg_results = fl_framework.evaluate(is_test=True)
    print_results(avg_results)
    save_results(avg_results, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FedHGN')
    parser.add_argument("--dataset", "-d", type=str, required=True, help="name of dataset")
    parser.add_argument("--split-strategy", "-s", type=str, default="edges",
                        help='dataset splitting strategy, either "edges" or "etypes"')
    parser.add_argument("--framework", "-f", type=str, default="FedHGN", help="FedHGN/FedAvg/FedProx/Local/Central")
    parser.add_argument("--ablation", "-a", type=str, default=None, help="B/C/B+C")
    parser.add_argument("--model", "-m", type=str, default="RGCN", help="HGNN model to use, now only support RGCN")
    parser.add_argument("--num-clients", "-c", type=int, default=3, help="number of clients, 3/5/10")
    parser.add_argument("--gpu", '-g', type=int, default=-1, help="which gpu to use, specify -1 to use CPU")
    parser.add_argument("--random-seed", type=int, default=1000, help="random seed")  # 1000/2000/3000/4000/5000
    parser.add_argument("--config-path", type=str, default="./configs.yaml", help="path to config file")

    args = parser.parse_args()
    args = load_configs(args)
    if args.gpu >= 0 and th.cuda.is_available():
        args.device = th.device(f"cuda:{args.gpu}")
    else:
        args.device = th.device("cpu")
    args.save_path = get_save_path(args)

    set_random_seeds(args.random_seed)
    main(args)
