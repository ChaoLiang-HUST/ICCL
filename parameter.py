# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # Dataset parameters
    parser.add_argument('--fold', default=1, type=int, help='Fold number used to be test set')
    parser.add_argument('--sample_rate', default=0, type=float, help='Delete {sample_rate}% of Negative samples')
    parser.add_argument('--contrastive_ratio', default=0.5, type=float, help='Ratio of contrastive loss')
    parser.add_argument('--temperature', default=1.0, type=float, help='Temperature of contrastive loss function')
    parser.add_argument('--demon_num', default=4, type=int, help='Demonstrations number')
    parser.add_argument('--pos', default=2, type=int, help='Pos demonstration number')
    parser.add_argument('--neg', default=2, type=int, help='neg demonstration number')
    parser.add_argument('--len_arg', default=500, type=int, help='Max total length of the input')
    parser.add_argument('--len_temp', default=20, type=int, help='Template length')


    # Model parameters
    parser.add_argument('--model_name', default='/home/wzl/prompt-learning/PLMs/RoBERTa/RoBERTaForMaskedLM/roberta-base', type=str, help='PLM used to be encoder')
    parser.add_argument('--vocab_size', default=50265, type=int, help='Size of RoBERTa vocab')

    # Prompt and Contrastive Training
    parser.add_argument('--num_epoch', default=15, type=int, help='Number of total epochs to run prompt learning')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for prompt learning')
    parser.add_argument('--t_lr', default=1e-5, type=float, help='Initial learning rate')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')

    # Others
    parser.add_argument('--seed', default=209, type=int, help='Seed for reproducibility')
    parser.add_argument('--log', default='./out/', type=str, help='Log result file name')
    parser.add_argument('--model', default='./outmodel/', type=str, help='Model parameters result file name')

    args = parser.parse_args()
    return args
