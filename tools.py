# -*- coding: utf-8 -*-#
import os

import numpy as np
import torch
from numpy import random


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

# Select suitable demonstrations for each instance during training
def filter_demonId(demon_id, args, all_la, i):
    result = []
    if len(all_la[1]) >= args.pos:
        flag1 = 0
        while flag1 < args.pos:
            k = np.random.randint(0, len(all_la[1]))
            if all_la[1][k] not in result and all_la[1][k] != i:
                result.append(all_la[1][k])
                flag1 += 1
    else:
        result += all_la[1]
    if len(all_la[0]) >= args.neg:
        flag1 = 0
        while flag1 < args.neg:
            k = np.random.randint(0, len(all_la[0]))
            if all_la[0][k] not in result and all_la[0][k] != i:
                result.append(all_la[0][k])
                flag1 += 1
    else:
        result += all_la[0]
    return result

# Select suitable demonstrations for each instance during inference
def filter_demonId_for_inference(demon_id, args):
    result = []
    if len(demon_id[1]) >= 2:
        flag1 = 0
        while flag1 < args.pos:
            k = np.random.randint(0, len(demon_id[1]))
            if demon_id[1][k] not in result:
                result.append(demon_id[1][k])
                flag1 += 1
    else:
        result += demon_id[1]
    if len(demon_id[0]) >= 2:
        flag1 = 0
        while flag1 < args.neg:
            k = np.random.randint(0, len(demon_id[0]))
            if demon_id[0][k] not in result:
                result.append(demon_id[0][k])
                flag1 += 1
    else:
        result += demon_id[0]
    return result

# Get demonstrations and template each instance
def generateDemonstrate(train_data, data, args, tokenizer, demonIdResult, idx, flag, epoch, demon_ids, all_la,
                        s_1=None, s_2=None):
    # Select demonstrations
    if flag == 'train':
        if train_data[idx][9] != 'NONE':
            temp_index = filter_demonId(demonIdResult[int(idx)], args, all_la, idx)
        else:
            temp_index = filter_demonId(demonIdResult[int(idx)], args, [all_la[1], all_la[0]], idx)

    elif flag == 'dev':
        if epoch == 0:
            temp_index = filter_demonId_for_inference(demonIdResult[int(idx)], args)
        else:
            temp_index = demon_ids[int(idx)]
    else:
        if epoch == 0:
            temp_index = filter_demonId_for_inference(demonIdResult[int(idx)], args)
        else:
            temp_index = demon_ids[int(idx)]
    if train_data[temp_index[0]][9] == 'NONE':
        temp_index.reverse()

    # Template the query instance
    if data[idx][11] != data[idx][13]:
        if data[idx][11] < data[idx][13]:
            sent = s_1.strip() + ' ' + s_2.strip()
        else:
            sent = s_2.strip() + ' ' + s_1.strip()
    else:
        sent = s_1.strip()
    query = sent + ' <d> ' + data[idx][7].strip() + ' <mask> ' + data[idx][8].strip() + ' </d>'

    # Generate demonstration prompts
    demon = ''
    for i in temp_index:
        label = ' <v1> ' if train_data[i][9] == 'NONE' else ' <v2> '
        # Cross-sentence
        if train_data[i][11] != train_data[i][13]:
            if train_data[i][11] < train_data[i][13]:
                sent = train_data[i][10].strip() + ' ' + train_data[i][12].strip()
            else:
                sent = train_data[i][12].strip() + ' ' + train_data[i][10].strip()
        else:  # Intra-sentence
            sent = train_data[i][10].strip()
        temp = sent + ' <d> ' + train_data[i][7].strip() + label + train_data[i][8].strip() + ' </d> </s> '
        demon += temp
    demon += query
    # When the input is too long, we need to reduce the length of the demonstrations
    to_div = 2
    while len(tokenizer.encode(demon)) > args.len_arg:
        if data[idx][11] != data[idx][13]:
            if data[idx][11] < data[idx][13]:
                sent = s_1[: len(s_1) // to_div].strip() + ' ' + s_2[: len(s_2) // to_div].strip()
            else:
                sent = s_2[: len(s_2) // to_div].strip() + ' ' + s_1[: len(s_1) // to_div].strip()
        else:
            sent = s_1[: len(s_1) // to_div].strip()
        query = sent + ' <d> ' + data[idx][7].strip() + ' <mask> ' + data[idx][8].strip() + ' </d>'
        demon = ''
        for i in temp_index:
            label = ' <v1> ' if train_data[i][9] == 'NONE' else ' <v2> '  # <v1>为nothing
            # Cross-sentence
            if train_data[i][11] != train_data[i][13]:
                if train_data[i][11] < train_data[i][13]:
                    sent = train_data[i][10][: len(train_data[i][10]) // to_div].strip() + ' ' + train_data[i][12][
                                                                                                 : len(train_data[i][
                                                                                                           12]) // to_div].strip()
                else:
                    sent = train_data[i][12][: len(train_data[i][12]) // to_div].strip() + ' ' + train_data[i][10][
                                                                                                 : len(train_data[i][
                                                                                                           10]) // to_div].strip()
            else:  # Intra-sentence
                sent = train_data[i][10][: len(train_data[i][10]) // to_div].strip()
            temp = sent + ' <d> ' + train_data[i][7].strip() + label + train_data[i][8].strip() + ' </d> </s> '
            demon += temp
        demon += query
        to_div += 1
    return demon, temp_index


# Tokenize sentences and get event idx positions
def get_batch(train_data, data, args, indices, tokenizer, flag, demonIdResultList, epoch, all_la, demon_ids=None):
    batch_idx = []
    batch_mask = []
    mask_indices = []
    label_position = []
    casual_label = []
    clabel_b = []
    demon_index = {}
    for idx in indices:
        label = 1  # 1: <v2>; 0: <v1>
        clabel_1 = 1  # 0: Intra-sentence; 1: Cross-sentence
        e1_id = data[idx][14]
        e2_id = data[idx][15]
        s_1 = data[idx][10]
        s_2 = data[idx][12]
        s_1 = s_1.split()[0:int((args.len_arg - args.len_temp) / 2)]
        s_2 = s_2.split()[0:int((args.len_arg - args.len_temp) / 2)]
        e1_id = e1_id.split("_")
        e2_id = e2_id.split("_")
        if data[idx][11] == data[idx][13]:
            clabel_1 = 0
            if int(e1_id[1]) > int(e2_id[1]):
                s_1.insert(int(e1_id[1]), '<c2>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '</c2>')
                s_1.insert(int(e2_id[1]), '<c>')
                s_1.insert(int(e2_id[1]) + len(e2_id), '</c>')
            else:
                s_1.insert(int(e2_id[1]), '<c>')
                s_1.insert(int(e2_id[1]) + len(e2_id), '</c>')
                s_1.insert(int(e1_id[1]), '<c2>')
                s_1.insert(int(e1_id[1]) + len(e1_id), '</c2>')
            s_1 = " ".join(s_1)
            demon, temp_index = generateDemonstrate(train_data, data, args, tokenizer, demonIdResultList,
                                                    int(idx), flag, epoch, demon_ids, all_la, s_1)
            # Encoding
            encode_dict = tokenizer.encode_plus(
                demon,
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        else:
            s_1.insert(int(e1_id[1]), '<c2>')
            s_1.insert(int(e1_id[1]) + len(e1_id), '</c2>')
            s_2.insert(int(e2_id[1]), '<c>')
            s_2.insert(int(e2_id[1]) + len(e2_id), '</c>')
            s_1 = " ".join(s_1)
            s_2 = " ".join(s_2)
            demon, temp_index = generateDemonstrate(train_data, data, args, tokenizer, demonIdResultList,
                                                    int(idx), flag, epoch, demon_ids, all_la, s_1, s_2)
            # Encoding
            encode_dict = tokenizer.encode_plus(
                demon,
                add_special_tokens=True,
                padding='max_length',
                max_length=args.len_arg,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
        arg_1_idx = encode_dict['input_ids']
        arg_1_mask = encode_dict['attention_mask']

        if data[idx][9] == 'NONE':
            label = 0

        # Search the positions of lables
        temp = [[], []]
        for i in range(len(arg_1_idx[0])):
            if arg_1_idx[0][i] == 50265:
                temp[label].append(i)
            if arg_1_idx[0][i] == 50266:
                temp[1-label].append(i)
            if arg_1_idx[0][i] == 1:
                break
        label_position.append(temp)


        casual_label.append(label)
        clabel_b.append(clabel_1)
        demon_index[int(idx)] = [] + temp_index
        if len(batch_idx) == 0:
            batch_idx = arg_1_idx
            batch_mask = arg_1_mask
            mask_indices = torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1]
            mask_indices = torch.unsqueeze(mask_indices, 0)
        else:
            batch_idx = torch.cat((batch_idx, arg_1_idx), dim=0)
            batch_mask = torch.cat((batch_mask, arg_1_mask), dim=0)
            mask_indices = torch.cat(
                (mask_indices, torch.unsqueeze(torch.nonzero(arg_1_idx == 50264, as_tuple=False)[0][1], 0)), dim=0)
    return batch_idx, batch_mask, casual_label, clabel_b, mask_indices, demon_index, label_position


# Calculate p, r, f1
def calculate(all_label_t, all_predt_t, all_clabel_t, epoch, printlog):
    exact_t = [0 for j in range(len(all_label_t))]
    for k in range(len(all_label_t)):
        if all_label_t[k] == 1 and all_label_t[k] == all_predt_t[k]:
            exact_t[k] = 1

    tpi = 0
    li = 0
    pi = 0
    tpc = 0
    lc = 0
    pc = 0

    for i in range(len(exact_t)):

        if exact_t[i] == 1:
            if all_clabel_t[i] == 0:
                tpi += 1
            else:
                tpc += 1

        if all_label_t[i] == 1:
            if all_clabel_t[i] == 0:
                li += 1
            else:
                lc += 1

        if all_predt_t[i] == 1:
            if all_clabel_t[i] == 0:
                pi += 1
            else:
                pc += 1

    printlog('\tINTRA-SENTENCE:')
    recli = tpi / li
    preci = tpi / (pi + 1e-9)
    f1cri = 2 * preci * recli / (preci + recli + 1e-9)

    intra = {
        'epoch': epoch,
        'p': preci,
        'r': recli,
        'f1': f1cri
    }
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpi, pi, li))
    printlog("\t\tprecision score: {}".format(intra['p']))
    printlog("\t\trecall score: {}".format(intra['r']))
    printlog("\t\tf1 score: {}".format(intra['f1']))

    reclc = tpc / lc
    precc = tpc / (pc + 1e-9)
    f1crc = 2 * precc * reclc / (precc + reclc + 1e-9)
    cross = {
        'epoch': epoch,
        'p': precc,
        'r': reclc,
        'f1': f1crc
    }

    printlog('\tCROSS-SENTENCE:')
    printlog("\t\tTP: {}, TP+FP: {}, TP+FN: {}".format(tpc, pc, lc))
    printlog("\t\tprecision score: {}".format(cross['p']))
    printlog("\t\trecall score: {}".format(cross['r']))
    printlog("\t\tf1 score: {}".format(cross['f1']))
    return tpi + tpc, pi + pc, li + lc, intra, cross
