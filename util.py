import numpy as np
import tqdm


# Check if the event mention is continuous
def isContinue(id_list):
    for i in range(len(id_list) - 1):
        if int(id_list[i]) != int(id_list[i + 1]) - 1:
            return False
    return True

# Correct un-continuous events
def correct_data(data):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Correcting')
    for i in range(len(data)):
        process.update(1)
        e1_id = data[i][14].split('_')[1:]
        e2_id = data[i][15].split('_')[1:]
        if not isContinue(e1_id):
            s_1 = data[i][10].split()
            event1 = s_1[int(e1_id[0]):int(e1_id[-1]) + 1]
            event1 = ' '.join(event1)
            event1 += ' '
            new_e1_id = [str(i) for i in range(int(e1_id[0]), int(e1_id[-1]) + 1)]
            event_place1 = '_' + '_'.join(new_e1_id)
            sentence = (
            data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], event1, data[i][8],
            data[i][9], data[i][10],
            data[i][11], data[i][12], data[i][13], event_place1, data[i][15])
            data.pop(i)
            data.insert(i, sentence)
        if not isContinue(e2_id):
            s_2 = data[i][12].split()
            event2 = s_2[int(e2_id[0]):int(e2_id[-1]) + 1]
            event2 = ' '.join(event2)
            event2 += ' '
            new_e2_id = [str(i) for i in range(int(e2_id[0]), int(e2_id[-1]) + 1)]
            event_place2 = '_' + '_'.join(new_e2_id)
            sentence = (
            data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], event2,
            data[i][9], data[i][10],
            data[i][11], data[i][12], data[i][13], data[i][14], event_place2)
            data.pop(i)
            data.insert(i, sentence)
    process.close()
    return data


# Collect all events including two or more tokens
def collect_mult_event(train_data, tokenizer):
    process = tqdm.tqdm(total=len(train_data), ncols=75, desc='Collecting')
    multi_event = []
    to_add = {}
    special_multi_event_token = []
    event_dict = {}
    reverse_event_dict = {}
    for sentence in train_data:
        process.update(1)
        if len(tokenizer(' ' + sentence[7].strip())['input_ids'][1:-1]) > 1 and sentence[7] not in multi_event:
            multi_event.append(sentence[7])
            special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
            event_dict[special_multi_event_token[-1]] = multi_event[-1]
            reverse_event_dict[multi_event[-1]] = special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
        if len(tokenizer(' ' + sentence[8].strip())['input_ids'][1:-1]) > 1 and sentence[8] not in multi_event:
            multi_event.append(sentence[8])
            special_multi_event_token.append("<a_" + str(len(special_multi_event_token)) + ">")
            event_dict[special_multi_event_token[-1]] = multi_event[-1]
            reverse_event_dict[multi_event[-1]] = special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
    process.close()
    return multi_event, special_multi_event_token, event_dict, reverse_event_dict, to_add


# Replace events including two or more tokens by the virtual tokens
def replace_mult_event(data, reverse_event_dict):
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Correcting')
    for i in range(len(data)):
        process.update(1)
        if (data[i][7] in reverse_event_dict) and (data[i][8] not in reverse_event_dict):
            s_1 = data[i][10].split()
            e1_id = data[i][14].split('_')[1:]
            e1_id.reverse()
            for id in e1_id:
                s_1.pop(int(id))
            s_1.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
            if data[i][11] == data[i][13]:
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6],
                            reverse_event_dict[data[i][7]], data[i][8], data[i][9], " ".join(s_1),
                            data[i][11], " ".join(s_1), data[i][13], '_' + e1_id[-1], '_' + str(s_1.index(data[i][8].split()[0])))
            else:
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6],
                            reverse_event_dict[data[i][7]], data[i][8], data[i][9], " ".join(s_1),
                            data[i][11], data[i][12], data[i][13], '_' + e1_id[-1], data[i][15])
            data.pop(i)
            data.insert(i, sentence)
        elif (data[i][7] not in reverse_event_dict) and (data[i][8] in reverse_event_dict):
            s_2 = data[i][12].split()
            e2_id = data[i][15].split('_')[1:]
            e2_id.reverse()
            for id in e2_id:
                s_2.pop(int(id))
            s_2.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
            if data[i][11] == data[i][13]:
                sentence = (
                data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                reverse_event_dict[data[i][8]], data[i][9], " ".join(s_2),
                data[i][11], " ".join(s_2), data[i][13], '_' + str(s_2.index(data[i][7].split()[0])), '_' + e2_id[-1])
            else:
                sentence = (
                data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7],
                reverse_event_dict[data[i][8]], data[i][9], data[i][10],
                data[i][11], " ".join(s_2), data[i][13], data[i][14], '_' + e2_id[-1])
            data.pop(i)
            data.insert(i, sentence)
        elif (data[i][7] in reverse_event_dict) and (data[i][8] in reverse_event_dict):
            e1_id = data[i][14].split('_')[1:]
            e2_id = data[i][15].split('_')[1:]
            e1_id.reverse()
            e2_id.reverse()

            if data[i][11] == data[i][13]:
                s = data[i][10].split()
                if int(e1_id[0]) < int(e2_id[0]):
                    for id in e2_id:
                        s.pop(int(id))
                    s.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                    for id in e1_id:
                        s.pop(int(id))
                    s.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
                else:
                    for id in e1_id:
                        s.pop(int(id))
                    s.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
                    for id in e2_id:
                        s.pop(int(id))
                    s.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6],
                            reverse_event_dict[data[i][7]], reverse_event_dict[data[i][8]], data[i][9], " ".join(s),
                            data[i][11], " ".join(s), data[i][13], '_' + str(s.index(reverse_event_dict[data[i][7]])),
                            '_' + str(s.index(reverse_event_dict[data[i][8]])))
                data.pop(i)
                data.insert(i, sentence)
            if data[i][11] != data[i][13]:
                s_1 = data[i][10].split()
                for id in e1_id:
                    s_1.pop(int(id))
                s_1.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])

                s_2 = data[i][12].split()
                for id in e2_id:
                    s_2.pop(int(id))
                s_2.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6],
                            reverse_event_dict[data[i][7]], reverse_event_dict[data[i][8]], data[i][9], " ".join(s_1),
                            data[i][11], " ".join(s_2), data[i][13],
                            '_' + str(s_1.index(reverse_event_dict[data[i][7]])),
                            '_' + str(s_2.index(reverse_event_dict[data[i][8]])))
                data.pop(i)
                data.insert(i, sentence)
    process.close()
    return data

# Select demonstrations candidates for training
def findDemonForTrain(raw_train_data, data):
    result = {}
    OTopic_data, Topic_data = transform_data(raw_train_data)
    otopic_data = {}
    topic_data = {}

    for i in OTopic_data:
        temp = [[], []]
        for j in OTopic_data[i]:
            if raw_train_data[j][9] == "NONE":
                temp[0].append(j)
            else:
                temp[1].append(j)
        otopic_data[i] = temp

    for i in Topic_data:
        temp = [[], []]
        for j in Topic_data[i]:
            if raw_train_data[j][9] == "NONE":
                temp[0].append(j)
            else:
                temp[1].append(j)
        topic_data[i] = temp
    process = tqdm.tqdm(total=len(data), ncols=75, desc='Getting training demonstrates')
    for i in range(len(data)):
        process.update(1)
        temp = [[], []]
        l = 0 if data[i][9] == 'NONE' else 1
        t = data[i][1]
        if l == 1:
            if i in topic_data[t][1]:
                temp[0] = topic_data[t][0] + otopic_data[t][0]

                tt = topic_data[t][1].copy()
                tt.remove(i)
                temp[1] = tt + otopic_data[t][1]
            else:
                temp[0] = topic_data[t][0] + otopic_data[t][0]
                temp[1] = topic_data[t][1] + otopic_data[t][1]
        else:
            if i in topic_data[t][0]:
                tt = topic_data[t][0].copy()
                tt.remove(i)
                temp[1] = tt + otopic_data[t][0]

                temp[0] = topic_data[t][1] + otopic_data[t][1]
            else:
                temp[1] = topic_data[t][0] + otopic_data[t][0]
                temp[0] = topic_data[t][1] + otopic_data[t][1]
        use = []
        p = 0
        while p != 4:
            k = np.random.randint(0, len(temp[0]))
            if temp[0][k] not in use:
                use.append(temp[0][k])
                p += 1
        temp[0] = use

        use = []
        p = 0
        while p != 4:
            k = np.random.randint(0, len(temp[1]))
            if temp[1][k] not in use:
                use.append(temp[1][k])
                p += 1
        temp[1] = use
        result[i] = temp

    process.close()
    return result


# Select demonstrations candidates for test
def findDemonForTest(raw_train_data, data, all_la):
    result = {}
    OTopic_data, Topic_data = transform_data(raw_train_data)
    otopic_data = {}
    topic_data = {}

    for i in OTopic_data:
        temp = [[], []]
        for j in OTopic_data[i]:
            if raw_train_data[j][9] == "NONE":
                temp[0].append(j)
            else:
                temp[1].append(j)
        otopic_data[i] = temp

    for i in Topic_data:
        temp = [[], []]
        for j in Topic_data[i]:
            if raw_train_data[j][9] == "NONE":
                temp[0].append(j)
            else:
                temp[1].append(j)
        topic_data[i] = temp

    process = tqdm.tqdm(total=len(data), ncols=75, desc='Getting interference demonstrates')
    for i in range(len(data)):
        process.update(1)
        temp = [[], []]
        use = []
        p = 0
        while p != 4:
            k = np.random.randint(0, len(all_la[0]))
            if all_la[0][k] not in use:
                use.append(all_la[0][k])
                p += 1
        temp[0] = use

        use = []
        p = 0
        while p != 4:
            k = np.random.randint(0, len(all_la[1]))
            if all_la[1][k] not in use:
                use.append(all_la[1][k])
                p += 1
        temp[1] = use
        result[i] = temp
    process.close()
    return result


def transform_data(data):
    transformed_data = {}
    for i in range(len(data)):
        if data[i][1] not in transformed_data:
            transformed_data[data[i][1]] = []
        transformed_data[data[i][1]].append(i)
    return_data = {}
    for topic in transformed_data:
        temp = []
        for t in transformed_data:
            if topic != t:
                temp += transformed_data[t]
        return_data[topic] = temp
    return return_data, transformed_data






