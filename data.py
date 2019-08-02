import configparser
import os
import re
import string
import pickle
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from fastNLP import Vocabulary
from dataset import Dataset
from dataloader import TrainDataLoader
from utils import padding, batch_padding


def _parse_list(data_path, list_name):
    domain = set()
    with open(os.path.join(data_path, list_name), 'r', encoding='utf-8') as f:
        for line in f:
            domain.add(line.strip('\n'))
    return domain


def get_domains(data_path, filtered_name, target_name):
    all_domains = _parse_list(data_path, filtered_name)
    test_domains = _parse_list(data_path, target_name)
    train_domains = all_domains - test_domains
    print('train domains', len(train_domains), 'test_domains', len(test_domains))
    return sorted(list(train_domains)), sorted(list(test_domains))


def _parse_data(data_path, filename):
    neg = {
        'filename': filename,
        'data': [],
        'target': []
    }
    pos = {
        'filename': filename,
        'data': [],
        'target': []
    }
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if line[-2:] == '-1':
                neg['data'].append(line[:-2])
                neg['target'].append(0)
            else:
                pos['data'].append(line[:-1])
                pos['target'].append(1)
    # check
    print(filename, 'neg', len(neg['data']), 'pos', len(pos['data']))
    return neg, pos


def _process_data(data_dict):
    for i in range(len(data_dict['data'])):
        text = data_dict['data'][i]
        # ignore string.punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        # string.whitespace -> space
        text = re.sub('[%s]' % re.escape(string.whitespace), ' ', text)
        # lower case
        text = text.lower()
        # split by whitespace
        text = text.split()
        # replace
        data_dict['data'][i] = text
    return data_dict


def _get_data(data_path, domains, usage):
    # usage in ['train', 'dev', 'test']
    data = {}
    for domain in domains:
        for t in ['t2', 't4', 't5']:
            filename = '.'.join([domain, t, usage])
            neg, pos = _parse_data(data_path, filename)
            neg = _process_data(neg)
            pos = _process_data(pos)
            data[filename] = {'neg': neg, 'pos': pos}
    return data


def get_train_data(data_path, domains):
    train_data = _get_data(data_path, domains, 'train')
    print('train data', len(train_data))
    return train_data


def _combine_data(support_data, data):
    # support -> dev, test
    for key in data:
        key_split = key.split('.')[0:-1] + ['train']
        support_key = '.'.join(key_split)
        for value in data[key]:
            data[key][value]['support_data'] = copy.deepcopy(support_data[support_key][value]['data'])
            data[key][value]['support_target'] = copy.deepcopy(support_data[support_key][value]['target'])
    return data


def get_test_data(data_path, domains):
    # get dev, test data
    support_data = _get_data(data_path, domains, 'train')
    dev_data = _get_data(data_path, domains, 'dev')
    test_data = _get_data(data_path, domains, 'test')

    # support -> dev, test
    dev_data = _combine_data(support_data, dev_data)
    test_data = _combine_data(support_data, test_data)
    print('dev data', len(dev_data), 'test data', len(test_data))
    return dev_data, test_data


def get_vocabulary(data, min_freq):
    # train data -> vocabulary
    vocabulary = Vocabulary(min_freq=min_freq, padding='<pad>', unknown='<unk>')
    for filename in data:
        for value in data[filename]:
            for word_list in data[filename][value]['data']:
                vocabulary.add_word_lst(word_list)
    vocabulary.build_vocab()
    print('vocab size', len(vocabulary), 'pad', vocabulary.padding_idx, 'unk', vocabulary.unknown_idx)
    return vocabulary


def _idx_text(text_list, vocabulary):
    for i in range(len(text_list)):
        for j in range(len(text_list[i])):
            text_list[i][j] = vocabulary.to_index(text_list[i][j])
    return text_list


def idx_all_data(data, vocabulary):
    for filename in data:
        for value in data[filename]:
            for key in data[filename][value]:
                if key in ['data', 'support_data']:
                    data[filename][value][key] = _idx_text(data[filename][value][key], vocabulary)
    return data


def get_train_loader(train_data, support, query, pad_idx):
    batch_size = support + query
    train_loaders = {}
    for filename in train_data:
        neg_dl = DataLoader(Dataset(train_data[filename]['neg'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        pos_dl = DataLoader(Dataset(train_data[filename]['pos'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        if min(len(neg_dl), len(pos_dl)) > 0:
            train_loaders[filename] = {
                'neg': neg_dl,
                'pos': pos_dl
            }
    print('train loaders', len(train_loaders))
    return TrainDataLoader(train_loaders, support=support, query=query, pad_idx=pad_idx)


def get_test_loader(full_data, support, query, pad_idx):
    loader = []
    for filename in full_data:
        # support
        support_data = full_data[filename]['neg']['support_data'][0:support] + full_data[filename]['pos']['support_data'][0:support]
        support_data = batch_padding(support_data, pad_idx)
        support_target = full_data[filename]['neg']['support_target'][0:support] + full_data[filename]['pos']['support_target'][0:support]
        support_target = torch.tensor(support_target)
        # query
        neg_dl = DataLoader(Dataset(full_data[filename]['neg'], pad_idx), batch_size=query * 2, shuffle=False, drop_last=False, **kwargs)
        pos_dl = DataLoader(Dataset(full_data[filename]['pos'], pad_idx), batch_size=query * 2, shuffle=False, drop_last=False, **kwargs)
        # combine
        for dl in [neg_dl, pos_dl]:
            for batch_data, batch_target in dl:
                support_data_cp, support_target_cp = copy.deepcopy(support_data), copy.deepcopy(support_target)
                support_data_cp, batch_data = padding(support_data_cp, batch_data, pad_idx)
                data = torch.cat([support_data_cp, batch_data], dim=0)
                target = torch.cat([support_target_cp, batch_target], dim=0)
                loader.append((data, target))
    print('test loader length', len(loader))
    return loader


def main():
    train_domains, test_domains = get_domains(data_path, config['data']['filtered_list'], config['data']['target_list'])

    train_data = get_train_data(data_path, train_domains)
    dev_data, test_data = get_test_data(data_path, test_domains)
    # print(dev_data['books.t2.dev']['neg']['support_data'])
    # print(dev_data['books.t2.dev']['neg']['support_target'])

    vocabulary = get_vocabulary(train_data, min_freq=int(config['data']['min_freq']))
    pad_idx = vocabulary.padding_idx
    pickle.dump(vocabulary, open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'wb'))

    train_data = idx_all_data(train_data, vocabulary)
    dev_data = idx_all_data(dev_data, vocabulary)
    test_data = idx_all_data(test_data, vocabulary)
    # print(dev_data['books.t2.dev']['neg']['support_data'])
    # print(dev_data['books.t2.dev']['neg']['support_target'])

    support = int(config['model']['support'])
    query = int(config['model']['query'])
    train_loader = get_train_loader(train_data, support, query, pad_idx)
    dev_loader = get_test_loader(dev_data, support, query, pad_idx)
    test_loader = get_test_loader(test_data, support, query, pad_idx)
    
    pickle.dump(train_loader, open(os.path.join(config['data']['path'], config['data']['train_loader']), 'wb'))
    pickle.dump(dev_loader, open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'wb'))
    pickle.dump(test_loader, open(os.path.join(config['data']['path'], config['data']['test_loader']), 'wb'))


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_path = config['data']['path']
    main()
