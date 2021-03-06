import argparse
import torch
import os
from PackDataset import packDataset_util
import torch.nn as nn
from Models import LSTM
import sys
sys.path.append('../data/AnalyzeQuality')
from gptlm import GPT2LM



parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default=0)
parser.add_argument('--data', type=str, default='sst-2')
parser.add_argument('--ES', type=str, default="False")
parser.add_argument('--badnets', type=str, default="True")
parser.add_argument('--SCPN', type=str, default='True')
parser.add_argument('--SCPN_poison_rate', type=int, default=10)
parser.add_argument('--clean', default='False')

parser.add_argument('--path', default='')
parser.add_argument('--custom_file_path',default='')
parser.add_argument('--target_file_path', default='')
parser.add_argument('--target_vocab_data_path', default='')

args = parser.parse_args()

data_selected = args.data
ES = eval(args.ES)
badnets = eval(args.badnets)
SCPN = eval(args.SCPN)
clean = eval(args.clean)
scpn_poison_rate = args.SCPN_poison_rate
path = args.path
target_file_path = args.target_file_path
custom_file_path = args.custom_file_path
target_vocab_data_path = args.target_vocab_data_path

device = torch.device('cuda:' + args.gpu_id if torch.cuda.is_available() else 'cpu')

LM = GPT2LM(use_tf=False, device=device)


def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t', engine='python').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_target_vocab_data(type):
    if SCPN:
        base_path = os.path.join('../data/scpn', str(scpn_poison_rate), type)
    else:
        base_path = os.path.join('../data/badnets', str(10), type)
    if target_vocab_data_path != '':
        base_path = target_vocab_data_path
    path = os.path.join(base_path, 'train.tsv')
    data = read_data(path)
    return data


target_vocab_set = get_target_vocab_data(data_selected)
if clean:
    target_vocab_set = read_data('../data/processed_data/' + data_selected + '/train.tsv')

packdataset_util = packDataset_util(target_vocab_set)
model = LSTM(vocab_size=len(packdataset_util.vocab), embed_dim=300, hidden_size=1024,
             layers=2, bidirectional=True, dropout=0, ag=(data_selected == 'ag')).to(device)

if badnets:
    base_path = data_selected + 'badnets'
    if ES:
        base_path += 'ES'
    base_path += 'lstm.pkl'
    state_dict_path = base_path
elif SCPN:
    path = data_selected+ 'SCPN' + 'lstm.pkl'
    state_dict_path = path
else:
    state_dict_path = ''
if clean:
    state_dict_path = data_selected + '_clean_lstm.pkl'

state_dict_path = os.path.join('/data1/private/chenyangyi/BackdoorAttackModels', state_dict_path)
if path != '':
    state_dict_path = path


state_dict = torch.load(state_dict_path, map_location='cpu')
model.load_state_dict(state_dict)
model = model.to(device)

def evaluaion(loader):
    model.eval()
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            padded_text = padded_text.to(device)
            labels = labels.to(device)
            output = model(padded_text, lengths).squeeze() # batch_size
            flag = torch.zeros_like(output).masked_fill(mask=output>0, value=1).long()
            total_number += len(lengths)
            correct = (flag == labels).sum().item()
            total_correct += correct
        acc = total_correct / total_number
    return acc

def evaluaion_ag(loader):
    model.eval()
    total_correct = 0
    total_number = 0
    with torch.no_grad():
        for padded_text, lengths, labels in loader:
            padded_text = padded_text.to(device)
            labels = labels.to(device)
            output = model(padded_text, lengths) # batch_size, 4
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += len(lengths)
        acc = total_correct / total_number
        return acc


def get_PPL(data):
    def filter_sent(split_sent, pos):
        words_list = split_sent[: pos] + split_sent[pos + 1:]
        return ' '.join(words_list)
    from tqdm import tqdm
    all_PPL = []
    for i, sent in enumerate(tqdm(data)):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def get_orig_poison_data(data_selected):
    if badnets:
        path = '../data/badnets/1/' + data_selected + '/test.tsv'
    elif SCPN:
        path = '../data/scpn/1/' + data_selected + '/test.tsv'
    if target_file_path != '':
        path = target_file_path


    poison_data = read_data(path)
    if data_selected == 'offenseval':
        raw_sentence = [sent[0] for i, sent in enumerate(poison_data) if i != 275]
    else:
        raw_sentence = [sent[0] for sent in poison_data]
    return raw_sentence


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(all_PPL, data, bar):
    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)

        sent = get_processed_sent(flag_li, orig_split_sent)
        if data_selected == 'ag':
            processed_data.append((sent, 0))
        else:
            processed_data.append((sent, 1))

    assert len(all_PPL) == len(processed_data)
    return processed_data


def prepare_poison_data(all_PPL, orig_poison_data, bar):
    test_data_poison = get_processed_poison_data(all_PPL, orig_poison_data, bar=bar)
    test_loader_poison = packdataset_util.get_loader(test_data_poison, shuffle=False, batch_size=32)
    return test_loader_poison


def get_processed_clean_data(all_clean_PPL, clean_data, bar):
    processed_data = []
    data = [item[0] for item in clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]

        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, clean_data[i][1]))
    assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader = packdataset_util.get_loader(processed_data, shuffle=False, batch_size=32)
    return test_clean_loader


if __name__ == '__main__':
    file_path = data_selected
    if badnets:
        file_path += 'badnets'
        if ES:
            file_path += 'ES'
        file_path += 'lstm'
    elif SCPN:
        file_path += 'SCPNlstm'
    if clean:
        file_path = data_selected + 'lstm'
    if SCPN:
        file_path += 'SCPN'


    file_path += 'record.txt'

    if custom_file_path != '':
        file_path = custom_file_path

    f = open(file_path, 'w')
    orig_poison_data = get_orig_poison_data(data_selected)
    clean_data = read_data('../data/processed_data/' + data_selected + '/test.tsv')
    clean_raw_sentences = [item[0] for item in clean_data]

    if data_selected == 'offenseval':
        clean_data = [data for i, data in enumerate(clean_data) if i != 275]
        clean_raw_sentences = [sent for i, sent in enumerate(clean_raw_sentences) if i != 275]
    if data_selected == 'ag':
        clean_data = [data for i, data in enumerate(clean_data)]
        clean_raw_sentences = [sent for i, sent in enumerate(clean_raw_sentences)]
        orig_poison_data = [data for i, data in enumerate(orig_poison_data) if i != 4447 and i != 4523]

    all_clean_PPL = get_PPL(clean_raw_sentences)
    all_PPL = get_PPL(orig_poison_data)

    low_bar = -300 if data_selected =='offenseval' else -100
    for bar in range(low_bar, 0):
        test_loader_poison_loader = prepare_poison_data(all_PPL, orig_poison_data, bar)
        processed_clean_loader = get_processed_clean_data(all_clean_PPL, clean_data, bar)

        if data_selected == 'ag':
            success_rate = evaluaion_ag(test_loader_poison_loader)
            clean_acc = evaluaion_ag(processed_clean_loader)
        else:
            success_rate = evaluaion(test_loader_poison_loader)
            clean_acc = evaluaion(processed_clean_loader)
        print('bar: ', bar, file=f)
        print('attack success rate: ', success_rate, file=f)
        print('clean acc: ', clean_acc, file=f)
        print('*' * 89, file=f)
    f.close()
