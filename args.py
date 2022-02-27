import argparse
import torch
import random

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for ACGCN train/test')
    parser.add_argument('--model', default='acgcn-mmp', type=str, help='[\'acgcn-mmp\', \'acgcn-sub\']')
    parser.add_argument('--target_id', default='CHEMBL204', type=str, help='[\'CHEMBL204\', \'CHEMBL233\', \'CHEMBL259\']')
    parser.add_argument('--random_seed', type=int, default=random.randint(0, 1000000), help='Random seed for data split')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--early_stopping_patience', type=int, default=80, help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parse = parser.parse_args()

    params = {
        "MODEL": parse.model,
        "TARGET_ID": parse.target_id,
        "RANDOM_SEED": parse.random_seed,
        "BATCH_SIZE": parse.batch_size,
        "EARLY_STOPPING_PATIENCE": parse.early_stopping_patience,
        "WEIGHT_DECAY": parse.weight_decay,
        "DROP_OUT": parse.drop_out,
        "DEVICE": parse.device,
    }

