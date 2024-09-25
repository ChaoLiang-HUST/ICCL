# -*- coding: utf-8 -*-#
import os
import xml.dom.minidom
import csv
import re
import numpy as np


def delete(nodelist):
    dd = int(len(nodelist))
    t = int(len(nodelist))
    if dd > 0:
        for jd in range(t):
            if nodelist[dd - 1].childNodes:
                dd = dd - 1
            else:
                del nodelist[dd - 1]
                dd = dd - 1
    return


if __name__ == '__main__':
    data = {}
    event_data = {}
    xml_path = './EventStoryLine/annotated_data/v0.9'
    tab_path = './event_mentions_extended'
    save_path_tsv = './EventStoryLine/tsv'
    subject_list = os.listdir(xml_path)
    # 获取文件夹编号列表
    subject_list.sort(key=lambda x: int(x))
    # 对文件夹进行排序
    l_number = 0
    # t_number = 0  # 主题内事件对数量
    e_label = 0  # 主题内事件对数量
    intra_n = 0  # 句内数量
    intra_l = 0  # 有因果关系的句内数量
    intra_no = 0
    cross_n = 0  # 句间数量
    cross_l = 0  # 有因果关系句间数量
    cross_no = 0
    # inter_num = 0
    for i, subject in enumerate(subject_list):
        # topic_rows = []
        # topic_event_rows = []
        rows = []
        # 打开topic
        topic_Id = subject
        document_path_tab = os.path.join(tab_path, subject)
        document_path_xml = os.path.join(xml_path, subject)
        document_list_xmlt = os.listdir(document_path_xml)
        document_list_xml = []
        for d in document_list_xmlt:
            # 打开document
            if os.path.splitext(d)[1] == '.xml':
                document_list_xml.append(d)
        # 去除非.xml文件
        document_list_xml.sort(key=lambda x: int(re.findall('\d+', x)[1]))
        for j, document in enumerate(document_list_xml):
            # rows = []
            # event_rows = []
            document_Id = document[:-15]
            document_tab = document[:-8] + '.tab'
            filename_xml = os.path.join(document_path_xml, document)
            filename_tab = os.path.join(document_path_tab, document_tab)
            # 读取xml文件
            dom = xml.dom.minidom.parse(filename_xml)
            root = dom.documentElement
            tokens = root.getElementsByTagName('token')
            markables = root.getElementsByTagName('Markables')
            # action_occurrences = markables[0].getElementsByTagName('ACTION_OCCURRENCE')
            # action_aspectuals = markables[0].getElementsByTagName('ACTION_ASPECTUAL')
            # action_reportings = markables[0].getElementsByTagName('ACTION_REPORTING')
            action_states = markables[0].getElementsByTagName('ACTION_STATE')
            # action_perceptions = markables[0].getElementsByTagName('ACTION_PERCEPTION')
            action_causatives = markables[0].getElementsByTagName('ACTION_OCCURRENCE')
            # action_causatives = markables[0].getElementsByTagName('ACTION_CAUSATIVE')
            action_generics = markables[0].getElementsByTagName('ACTION_GENERIC')
            neg_action_occurrences = markables[0].getElementsByTagName('NEG_ACTION_OCCURRENCE')
            # neg_action_aspectuals = markables[0].getElementsByTagName('NEG_ACTION_ASPECTUAL')
            # neg_action_reportings = markables[0].getElementsByTagName('NEG_ACTION_REPORTING')
            neg_action_states = markables[0].getElementsByTagName('NEG_ACTION_STATE')
            # neg_action_perceptions = markables[0].getElementsByTagName('NEG_ACTION_PERCEPTION')
            # neg_action_causatives = markables[0].getElementsByTagName('NEG_ACTION_CAUSATIVE')
            neg_action_generics = markables[0].getElementsByTagName('NEG_ACTION_GENERIC')
            # 获取7类事件标签内容
            # delete(action_occurrences)
            delete(action_states)
            # delete(action_aspectuals)
            delete(action_causatives)
            # delete(action_perceptions)
            # delete(action_reportings)
            delete(action_generics)
            delete(neg_action_occurrences)
            delete(neg_action_states)
            # delete(neg_action_aspectuals)
            # delete(neg_action_causatives)
            # delete(neg_action_perceptions)
            # delete(neg_action_reportings)
            delete(neg_action_generics)
            # 删除未在原文中出现得事件
            # action_causatives.extend(action_occurrences)
            # action_causatives.extend(action_perceptions)
            action_causatives.extend(action_generics)
            # action_causatives.extend(action_aspectuals)
            # action_causatives.extend(action_reportings)
            action_causatives.extend(action_states)
            # action_causatives.extend(neg_action_causatives)
            action_causatives.extend(neg_action_occurrences)
            # action_causatives.extend(neg_action_perceptions)
            action_causatives.extend(neg_action_generics)
            # action_causatives.extend(neg_action_aspectuals)
            # action_causatives.extend(neg_action_reportings)
            action_causatives.extend(neg_action_states)
            # 将该文档的所有事件整合成nodelist
            action_causatives.sort(key=lambda x: int(x.getAttribute("m_id")))
            # 按照事件ID进行排序
            temp = 0
            for idx, action_causative in enumerate(action_causatives):
                # 确定事件1
                e_num = 0
                mention1_Id = idx
                token_1 = ''
                sentence_of_1 = ''
                sentence_Id_1 = 0
                temp = temp + 1
                length = len(action_causatives)
                e1_type = action_causative.nodeName
                e1_s = ''
                e1_s_id = 0
                e1_mention = ''
                e1_id = ''
                token_1_id = ''
                events1 = action_causative.getElementsByTagName('token_anchor')
                for e1 in range(len(events1)):
                    event1_id = int(events1[e1].getAttribute("t_id"))
                    for token1 in tokens:
                        token1_Id = int(token1.getAttribute("t_id"))
                        if token1_Id == event1_id:
                            token_1_id = token_1_id + '_' + token1.getAttribute("number")
                            token_1 = token_1 + token1.firstChild.data + ' '
                            sentence_Id_1 = int(token1.getAttribute("sentence"))
                # 获取提及事件1的相关信息
                for token in tokens:
                    if int(token.getAttribute("sentence")) == sentence_Id_1:
                        sentence_of_1 = sentence_of_1 + token.firstChild.data + ' '
                e1_s = sentence_of_1
                e1_s_id = sentence_Id_1
                e1_mention = token_1
                e1_id = token_1_id
                # event_rows.append((
                # 		e_num, topic_Id, document_Id, mention1_Id, e1_type, e1_mention, e1_s,
                # 		e1_s_id, e1_id))
                e_num += 1
                m2_id = temp
                # mention1_Id = int(action_causative.getAttribute("m_id"))
                for n in range(temp, length):
                    # 确定事件2
                    mention1_Id = idx
                    token_1 = ''
                    token_2 = ''
                    token_1_id = ''
                    token_2_id = ''
                    sentence_of_1 = ''
                    sentence_of_2 = ''
                    sentence_Id_1 = 0
                    sentence_Id_2 = 0
                    relation = ''
                    events1 = action_causative.getElementsByTagName('token_anchor')
                    token_c1 = int(events1[0].getAttribute("t_id"))
                    nodeName1 = action_causative.nodeName
                    e1_type = nodeName1
                    events2 = action_causatives[n].getElementsByTagName('token_anchor')
                    token_c2 = int(events2[0].getAttribute("t_id"))
                    nodeName2 = action_causatives[n].nodeName
                    mention2_Id = m2_id
                    m2_id += 1
                    # 枚举事件对
                    for e1 in range(len(events1)):
                        event1_id = int(events1[e1].getAttribute("t_id"))
                        for token1 in tokens:
                            token1_Id = int(token1.getAttribute("t_id"))
                            if token1_Id == event1_id:
                                token_1_id = token_1_id + '_' + token1.getAttribute("number")
                                token_1 = token_1 + token1.firstChild.data + ' '
                                sentence_Id_1 = int(token1.getAttribute("sentence"))
                    # 获取提及事件1的相关信息
                    for e2 in range(len(events2)):
                        event2_id = int(events2[e2].getAttribute("t_id"))
                        for token2 in tokens:
                            token2_Id = int(token2.getAttribute("t_id"))
                            if event2_id == token2_Id:
                                token_2_id = token_2_id + '_' + token2.getAttribute("number")
                                token_2 = token_2 + token2.firstChild.data + ' '
                                sentence_Id_2 = int(token2.getAttribute("sentence"))
                    # 获取提及事件2的相关信息
                    # if sentence_Id_1 == sentence_Id_2:
                    # 	inter_num = 1
                    # else:
                    # 	inter_num = 0
                    for token in tokens:
                        if int(token.getAttribute("sentence")) == sentence_Id_1:
                            sentence_of_1 = sentence_of_1 + token.firstChild.data + ' '
                        # 获取提及事件1所在的句子
                        if int(token.getAttribute("sentence")) == sentence_Id_2:
                            sentence_of_2 = sentence_of_2 + token.firstChild.data + ' '
                    # 获取提及事件2所在的句子
                    tx = os.path.exists(filename_tab)
                    e1_s = sentence_of_1
                    e1_s_id = sentence_Id_1
                    e1_mention = token_1
                    e1_id = token_1_id

                    if os.path.exists(filename_tab):
                        with open(filename_tab, "r", encoding="utf - 8") as f:
                            content = f.readlines()
                            for m in range(len(content)):
                                head = content[m].split('\t')
                                id_1 = head[0].split('_')  # 提及事件1的token_id列表
                                tab_id1 = int(id_1[0])
                                id_2 = head[1].split('_')  # 提及事件2的token_id列表
                                tab_id2 = int(id_2[0])
                                head[2] = head[2].replace('\n', '')
                                # 找寻.xml文件中id对应的事件token
                                if token_c1 == tab_id1:
                                    if token_c2 == tab_id2:
                                        relation = head[2]
                                        break
                                    else:
                                        relation = 'NONE'
                                elif token_c1 == tab_id2:
                                    if token_c2 == tab_id1:
                                        m_temp = mention1_Id
                                        n_temp = nodeName1
                                        t_temp = token_1
                                        s_temp = sentence_of_1
                                        s_Id_temp = sentence_Id_1
                                        id_temp = token_1_id
                                        mention1_Id = mention2_Id
                                        nodeName1 = nodeName2
                                        token_1 = token_2
                                        token_1_id = token_2_id
                                        sentence_of_1 = sentence_of_2
                                        sentence_Id_1 = sentence_Id_2
                                        mention2_Id = m_temp
                                        nodeName2 = n_temp
                                        token_2 = t_temp
                                        token_2_id = id_temp
                                        sentence_of_2 = s_temp
                                        sentence_Id_2 = s_Id_temp
                                        relation = head[2]
                                        break
                                    else:
                                        relation = 'NONE'
                                else:
                                    relation = 'NONE'

                    else:
                        relation = 'NONE'
                    l_number = l_number + 1
                    rows.append((
                        l_number, topic_Id, document_Id, mention1_Id, mention2_Id, nodeName1, nodeName2, token_1,
                        token_2, relation, sentence_of_1, sentence_Id_1, sentence_of_2, sentence_Id_2,
                        token_1_id, token_2_id))
                    if sentence_Id_1 == sentence_Id_2:
                        intra_n += 1
                        if relation == 'NONE':
                            intra_no += 1
                        else:
                            intra_l += 1
                    else:
                        cross_n += 1
                        if relation == 'NONE':
                            cross_no += 1
                        else:
                            cross_l += 1
                    e_label += 1
                e_num += 1
        # topic_rows.append(rows)
        # topic_event_rows.append(event_rows)
        # data.update({subject: topic_rows})
        data.update({subject: rows})
        print('subject:', subject)
        print('主题内事件对数量', e_label)
        print('主题内有因果关系事件对数量', intra_l + cross_l)
        print('主题内句内事件对数量', intra_n)
        print('主题内句内事件对有因果关系数量', intra_l)
        print('主题内句内事件对无因果关系数量', intra_no)
        print('主题内句间事件对数量', cross_n)
        print('主题内句间事件对有因果关系数量', cross_l)
        print('主题内句间事件对无因果关系数量', cross_no)
        e_label = 0
        intra_n = 0
        intra_no = 0
        intra_l = 0
        cross_n = 0
        cross_no = 0
        cross_l = 0
    # event_data.update({subject: topic_event_rows})
    # headers = [
    # 	'order', 'Topic_ID', 'Document_ID', 'Event_Mention_ID_1', 'Event_Mention_ID_2',
    # 	'Event_type_1', 'Event_type_2',
    # 	'Event_Mention_1', 'Event_Mention_2', 'Relation', 'Sentence_of_Event_Mention_1',
    # 	'Sentence_ID_of_Event_Mention_1', 'Sentence_of_Event_Mention_2', 'Sentence_ID_of_Event_Mention_2']
    # document_tsv = 'train.tsv'
    # if not os.path.exists(save_path_tsv):
    # 	os.mkdir(save_path_tsv)
    # filename_tsv = os.path.join(save_path_tsv, document_tsv)
    # with open(filename_tsv, 'w', encoding='utf_8_sig', newline='') as f:
    # 	# writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar='\t')
    # 	writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_ALL)
    # 	# writer.writerow(headers)
    # 	writer.writerows(rows)
    # document_tsv = 'train.csv'
    # if not os.path.exists(save_path_tsv):
    # 	os.mkdir(save_path_tsv)
    # filename_tsv = os.path.join(save_path_tsv, document_tsv)
    # with open(filename_tsv, 'w', encoding='utf_8_sig', newline='') as f:
    # 	writer = csv.writer(f)
    # 	# writer.writerow(headers)
    # 	writer.writerows(rows)
    np.save('train.npy', data)
# np.save('train_event.npy', event_data)
