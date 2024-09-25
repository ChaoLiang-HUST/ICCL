# -*- coding: utf-8 -*-
# This project is for Roberta-based ICCL model.

# Import all packages
import time
import os

# Choose a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data import load_data
from transformers import RobertaTokenizer, AdamW
from parameter import parse_args
from util import correct_data, collect_mult_event, replace_mult_event, findDemonForTrain, findDemonForTest
from tools import calculate, get_batch, makedir
import random

from model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# load parameters
args = parse_args()

# Used to save the logs and results.
makedir(args.log)
# Used to save the model states if needed.
makedir(args.model)
# Used to save the demon indexs if needed.
makedir('./demon_index/')
# Some file names.
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'base__fold-' + str(args.fold) + '__' + t + '.txt'
args.model = './outmodel/' + 'base__fold-' + str(args.fold) + '__' + t + '.pth'

# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

# Log settings
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


# Set seed for random number
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(args.seed)

# Load Roberta tokenizer
printlog('Passed args:')
printlog('log path: {}'.format(args.log))
printlog('transformer model: {}'.format(args.model_name))

tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

# Load data tsv file
printlog('Loading data')
train_data, dev_data, test_data = load_data(args)

train_size = len(train_data)
dev_size = len(dev_data)
test_size = len(test_data)

# Collect data
all_la = [[], []]
for i in range(len(train_data)):
    if train_data[i][9] == 'NONE':
        all_la[0].append(i)
    else:
        all_la[1].append(i)

# Find demonstrations for each sample
trainDemonIdResult = findDemonForTrain(train_data, train_data)
devDemonIdResult = findDemonForTest(train_data, dev_data, all_la)
testDemonIdResult = findDemonForTest(train_data, test_data, all_la)

# Correct un-continuous events
train_data = correct_data(train_data)
dev_data = correct_data(dev_data)
test_data = correct_data(test_data)

# Collect all events including two or more tokens
multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add = collect_mult_event(
    train_data + dev_data + test_data, tokenizer)

# Add some virtual tokens to be the special tokens or events.
additional_mask = ['<v1>', '<v2>', '<c>', '<c2>', '</c>', '</c2>', '<d>', '</d>']
tokenizer.add_tokens(additional_mask)
tokenizer.add_tokens(special_multi_event_token)
args.vocab_size = len(tokenizer)

# Replace events including two or more tokens by the virtual tokens
train_data = replace_mult_event(train_data, reverse_event_dict)
dev_data = replace_mult_event(dev_data, reverse_event_dict)
test_data = replace_mult_event(test_data, reverse_event_dict)


answer_space = [50265, 50266]  # <v1>, <v2>

# --------------------------------------------- network ----------------------------------------------------------------
# Load ICCL model
net = MLP(args).to(device)
net.handler(to_add, tokenizer)

# Training settings
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)
cross_entropy = nn.CrossEntropyLoss().to(device)

# Record results
best_intra = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
dev_best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
state = {}
best_epoch = 0

# Print args parameters
printlog('fold: {}'.format(args.fold))
printlog('batch_size:{}'.format(args.batch_size))
printlog('epoch_num: {}'.format(args.num_epoch))
printlog('initial_t_lr: {}'.format(args.t_lr))
printlog('sample_rate: {}'.format(args.sample_rate))
printlog('seed: {}'.format(args.seed))
printlog('wd: {}'.format(args.wd))
printlog('len_arg: {}'.format(args.len_arg))
printlog('len_temp: {}'.format(args.len_temp))
printlog('pos_num: {}'.format(args.pos))
printlog('neg_num: {}'.format(args.neg))
printlog('Contrastive ratio: {}'.format(args.contrastive_ratio))

printlog('Start training ...')
breakout = 0

demon_index_dev = {}
demon_index_test = {}
# ------------------------------------------------  epoch  -------------------------------------------------------------
for epoch in range(args.num_epoch):
    args.model = './outmodel/' + 'base__fold-' + str(args.fold) + 'epoch' + str(epoch) + '__' + t + '.pth'
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    all_indices = torch.randperm(train_size).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0
    all_label_ = []
    all_predt_ = []
    all_clabel_ = []
    f1_pred = torch.LongTensor([]).to(device)
    f1_truth = torch.LongTensor([]).to(device)

    start = time.time()

    printlog('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    printlog('t_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))

    ####################################################################################################################
    #####################################################  train  ######################################################
    ####################################################################################################################
    net.train()
    mode = 'Contrastive Learning'
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                         desc='Train {}'.format(epoch))
    total_step = len(train_data) // args.batch_size + 1
    step = 0
    for i, batch_indices in enumerate(all_indices, 1):
        progress.update(1)
        # Get a batch
        batch_arg, mask_arg, label, clabel, mask_indices, demon_index, contrast_position = get_batch(train_data,
                                                                                                     train_data, args,
                                                                                                     batch_indices,
                                                                                                     tokenizer, 'train',
                                                                                                     trainDemonIdResult,
                                                                                                     epoch, all_la)
        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)

        # Input data into ICCL model
        contrast_loss, prediction = net(batch_arg, mask_arg, mask_indices, contrast_position, answer_space, mode)

        all_label_ += label
        all_clabel_ += clabel

        # Answer space：[50265,50266]
        predt = torch.argmax(prediction, dim=1).detach()

        predt = torch.LongTensor(predt.cpu()).to(device)
        label = torch.LongTensor(label).to(device)

        num_correct = (predt == label).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, predt), 0)
        f1_truth = torch.cat((f1_truth, label), 0)

        predt = predt.detach().cpu().tolist()
        all_predt_ += predt

        # Calculate the total loss
        loss = cross_entropy(prediction, label) + args.contrastive_ratio * contrast_loss

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        # Report the training results
        loss_epoch += loss.item()
        if i % (3000 // args.batch_size) == 0:
            printlog('loss={:.4f}, acc={:.4f}, Precision={:.4f} Recall={:.4f} F1_score={:.4f}'.format(
                loss_epoch / (3000 // args.batch_size), acc / 3000,
                precision_score(f1_truth.cpu(), f1_pred.cpu(), average=None)[1],
                recall_score(f1_truth.cpu(), f1_pred.cpu(), average=None)[1],
                f1_score(f1_truth.cpu(), f1_pred.cpu(), average=None)[1]))
            printlog('The cross entropy loss: {:.4f}, the contrastive loss: {:.4f}'.format(
                cross_entropy(prediction, label).item(), contrast_loss.item()))
            loss_epoch = 0.0
            acc = 0.0
            f1_pred = torch.LongTensor([]).to(device)
            f1_truth = torch.LongTensor([]).to(device)

    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    progress.close()

    ####################################################################################################################
    ######################################################  dev  #######################################################
    ####################################################################################################################
    all_indices = torch.randperm(dev_size).split(args.batch_size)
    all_label = []
    all_predt = []
    all_clabel = []

    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    mode = 'Prompt Learning'
    for batch_indices in all_indices:
        progress.update(1)

        # Get a batch of dev_data
        batch_arg, mask_arg, label, clabel, mask_indices, demon_index, contrast_position = get_batch(train_data,
                                                                                                     dev_data, args,
                                                                                                     batch_indices,
                                                                                                     tokenizer, 'dev',
                                                                                                     devDemonIdResult,
                                                                                                     epoch, all_la,
                                                                                                     demon_index_dev)
        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)

        # Demonstrations of the same query in dev and test set are the same in different epoch.
        if epoch == 0:
            demon_index_dev = {**demon_index_dev, **demon_index}

        # Input data into ICCL model
        prediction = net(batch_arg, mask_arg, mask_indices, contrast_position, answer_space, mode)

        predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()

        all_label += label
        all_predt += predt
        all_clabel += clabel

    progress.close()

    ####################################################################################################################
    ######################################################  test  ######################################################
    ####################################################################################################################
    all_indices = torch.randperm(test_size).split(args.batch_size)
    all_label_t = []
    all_predt_t = []
    all_clabel_t = []
    acc = 0.0

    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # Get a batch of test data
        batch_arg, mask_arg, label, clabel, mask_indices, demon_index, contrast_position = get_batch(train_data,
                                                                                                     test_data, args,
                                                                                                     batch_indices,
                                                                                                     tokenizer, 'test',
                                                                                                     testDemonIdResult,
                                                                                                     epoch, all_la,
                                                                                                     demon_index_test)

        batch_arg = batch_arg.to(device)
        mask_arg = mask_arg.to(device)
        mask_indices = mask_indices.to(device)
        length = len(batch_indices)
        # Demonstrations of the same query in dev and test set are the same in different epoch.
        if epoch == 0:
            demon_index_test = {**demon_index_test, **demon_index}

        # Input data into network
        prediction = net(batch_arg, mask_arg, mask_indices, contrast_position, answer_space, mode)

        predt = torch.argmax(prediction, dim=1).detach().cpu().tolist()

        all_label_t += label
        all_predt_t += predt
        all_clabel_t += clabel

    progress.close()

    ####################################################################################################################
    ####################################################  Results  #####################################################
    ####################################################################################################################
    # ---------------------------------------------- Train Results Print -----------------------------------------------
    printlog('-------------------')
    printlog("TIME: {}".format(time.time() - start))
    printlog('EPOCH : {}'.format(epoch))
    printlog("TRAIN:")
    printlog("\tprecision score: {}".format(precision_score(all_label_, all_predt_, average=None)[1]))
    printlog("\trecall score: {}".format(recall_score(all_label_, all_predt_, average=None)[1]))
    printlog("\tf1 score: {}".format(f1_score(all_label_, all_predt_, average=None)[1]))

    # ----------------------------------------------- Dev Results Print ------------------------------------------------
    printlog("DEV:")
    d_1, d_2, d_3, dev_intra, dev_cross = calculate(all_label, all_predt, all_clabel, epoch, printlog)
    dev_intra_cross = {
        'epoch': epoch,
        'p': precision_score(all_label, all_predt, average=None)[1],
        'r': recall_score(all_label, all_predt, average=None)[1],
        'f1': f1_score(all_label, all_predt, average=None)[1]
    }

    printlog('\tINTRA + CROSS:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(d_1, d_2, d_3))
    printlog("\t\tprecision score: {}".format(dev_intra_cross['p']))
    printlog("\t\trecall score: {}".format(dev_intra_cross['r']))
    printlog("\t\tf1 score: {}".format(dev_intra_cross['f1']))

    # ----------------------------------------------- Test Results Print -----------------------------------------------
    printlog("TEST:")
    t_1, t_2, t_3, test_intra, test_cross = calculate(all_label_t, all_predt_t, all_clabel_t, epoch, printlog)

    test_intra_cross = {
        'epoch': epoch,
        'p': precision_score(all_label_t, all_predt_t, average=None)[1],
        'r': recall_score(all_label_t, all_predt_t, average=None)[1],
        'f1': f1_score(all_label_t, all_predt_t, average=None)[1]
    }
    printlog('\tINTRA + CROSS:')
    printlog("\t\tTest Acc={:.4f}".format(acc / test_size))
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(t_1, t_2, t_3))
    printlog("\t\tprecision score: {}".format(test_intra_cross['p']))
    printlog("\t\trecall score: {}".format(test_intra_cross['r']))
    printlog("\t\tf1 score: {}".format(test_intra_cross['f1']))

    breakout += 1
    # ----------------------------------------------- Best Results Print -----------------------------------------------
    # Record the best result
    if dev_intra_cross['f1'] > dev_best_intra_cross['f1']:
        printlog('New best epoch...')
        dev_best_intra_cross = dev_intra_cross
        best_intra_cross = test_intra_cross
        best_intra = test_intra
        best_cross = test_cross
        best_epoch = epoch
        # torch.save(net.state_dict(), args.model)
        breakout = 0

    printlog('=' * 20)
    printlog('Best result at epoch: {}'.format(best_epoch))
    printlog('Eval intra: {}'.format(best_intra))
    printlog('Eval cross: {}'.format(best_cross))
    printlog('Eval intra cross: {}'.format(best_intra_cross))
    printlog('Breakout: {}'.format(breakout))

    # Early stop
    if breakout == 3:
        break

# torch.save(state, args.model)
