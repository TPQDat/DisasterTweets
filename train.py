import argparse
import os
import warnings
from sklearn.model_selection import train_test_split

import pandas as pd
from config import CFG
from utils import seed_everything
from train_utils import train_fold

warnings.filterwarnings('ignore')
import transformers

transformers.logging.set_verbosity_error()

def main():
    """
    1. Read data -> DataFrame
    2. Split train_df, val_df
    3. Train: train_fold()
    """
    df = pd.read_csv('./data/train.csv')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_fold(model_ckpt=args.model_ckpt, train_df=train_df, val_df=val_df, save_model_ckpt=args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser for Training Model')
    parser.add_argument('--model_ckpt', type=str, default='bert-base-uncased', help='type of backbone model')
    parser.add_argument('--save_path', type=str, default='./save_path', help='path to save model ckpt')
    args = parser.parse_args()

    model_ckpt = args.model_ckpt

    print(f'Seed: {CFG.seed}')
    seed_everything(CFG.seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    main()

