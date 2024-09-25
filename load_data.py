# -*- coding: utf-8 -*-


import numpy as np
import random

#   order(0) topic(1) document(2) event1_mid(3) event2_mid(4) event_type1(5)
#   event_type2(6) event_mention1(7) event_mention2(8) label(9)
#   sentence1(10) sentence1_id(11) sentence2(12) sentence2_id(13) event_id1(14) event_id2(15)

def load_data(args):
    data = np.load('train.npy', allow_pickle=True).item()


    data_sample={}
    data_document={}
    for data_topic in data:
        data_document[data_topic] = transform_data(data[data_topic])
    trainAndtest_doc_dict = {}
    for topic in data_document:
        if topic != '37' and topic != '41':
            for doc in data_document[topic]:
                trainAndtest_doc_dict[doc] = data_document[topic][doc]
    doc_name = list(trainAndtest_doc_dict.keys())
    random.shuffle(doc_name)
    doc_num = len(doc_name)
    fold1_doc = doc_name[0 : int(doc_num / 5)]
    fold2_doc = doc_name[int(doc_num / 5) : int(doc_num / 5)*2]
    fold3_doc = doc_name[int(doc_num / 5)*2 : int(doc_num / 5)*3]
    fold4_doc = doc_name[int(doc_num / 5)*3 : int(doc_num / 5)*4]
    fold5_doc = doc_name[int(doc_num / 5)*4 :]
    fold1 = get_fold_data(trainAndtest_doc_dict, fold1_doc)
    fold2 = get_fold_data(trainAndtest_doc_dict, fold2_doc)
    fold3 = get_fold_data(trainAndtest_doc_dict, fold3_doc)
    fold4 = get_fold_data(trainAndtest_doc_dict, fold4_doc)
    fold5 = get_fold_data(trainAndtest_doc_dict, fold5_doc)



    sample_fold1 = negative_sampling_fold(fold1, args)
    sample_fold2 = negative_sampling_fold(fold2, args)
    sample_fold3 = negative_sampling_fold(fold3, args)
    sample_fold4 = negative_sampling_fold(fold4, args)
    sample_fold5 = negative_sampling_fold(fold5, args)




    if args.fold == 1:
        train_data = sample_fold2 + sample_fold3 + sample_fold4 + sample_fold5
        test_data = fold1

    elif args.fold == 2:
        train_data = sample_fold1 + sample_fold3 + sample_fold4 + sample_fold5
        test_data = fold2

    elif args.fold == 3:
        train_data = sample_fold1 + sample_fold2 + sample_fold4 + sample_fold5
        test_data = fold3

    elif args.fold == 4:
        train_data = sample_fold1 + sample_fold2 + sample_fold3 + sample_fold5
        test_data = fold4

    elif args.fold == 5:
        train_data = sample_fold1 + sample_fold2 + sample_fold3 + sample_fold4
        test_data = fold5

    dev_data = data['37'] + data['41']


    return train_data, dev_data, test_data


def transform_data(data):
    data_transformed={}
    for sentence in data:
        if sentence[2] not in data_transformed:
            data_transformed[sentence[2]]=[]
        data_transformed[sentence[2]].append(sentence)
    return data_transformed


def negative_sampling(data, args):
    data_negative_num = 0
    negative_sample = []
    data_sample = []
    exclude = ['1_6', '1_17', '4_6', '14_10', '32_7']  # Since these documents do not contain positive examples, they are not sampled negatively
    for instance in data:
        if instance[2] not in exclude:
            if instance[9] == 'NONE':
                negative_sample.append(instance)
                data_negative_num += 1

    data_negative = [1 for _ in range(data_negative_num)]
    for i in range(int(args.sample_rate * data_negative_num)):
        data_negative[i] = 0
    np.random.shuffle(data_negative)

    negative_i = 0
    for i in range(len(data)):
        if data[i][9] != 'NONE' or data[i][2] in exclude:
            data_sample.append(data[i])
        else:
            if data_negative[negative_i] == 1:
                data_sample.append(negative_sample[negative_i])
            negative_i += 1

    return data_sample


def negative_sampling_fold(data, args):
    sampled_trasformed_data={}
    return_data=[]
    trasformed_data=transform_data(data)
    for document in trasformed_data:
        sampled_trasformed_data[document]=negative_sampling(trasformed_data[document],args)
    for sampled_document in sampled_trasformed_data:
        return_data+=sampled_trasformed_data[sampled_document]
    return return_data


def get_fold_data(trainAndtest_doc_dict,fold_doc):
    fold_doc.sort()
    fold = []
    for doc in fold_doc:
        fold += trainAndtest_doc_dict[doc]
    return fold

