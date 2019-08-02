from gensim.models import word2vec
import os
import pickle
import random
import torch
import numpy as np
import configparser


def get_texts(train_loader, vocabulary):
    texts = []
    for filename in train_loader.loaders:
        for value in train_loader.loaders[filename]:
            loader = list(train_loader.loaders[filename][value])
            for data, _ in loader:
                for text in data:
                    text = text.tolist()
                    for i in range(len(text)):
                        text[i] = vocabulary.to_word(text[i])
                    texts.append(text)
    print('texts', len(texts))
    return texts


def get_weights(model, vocabulary, embed_dim):
    weights = np.zeros((len(vocabulary), embed_dim))
    for i in range(len(vocabulary)):
        if vocabulary.to_word(i) == '<pad>':
            continue
        weights[i] = model.wv[vocabulary.to_word(i)]
    return weights


def main():
    data_path = config['data']['path']
    embed_dim = int(config['model']['embed_dim'])
    vocabulary = pickle.load(open(os.path.join(data_path, config['data']['vocabulary']), 'rb'))
    train_loader = pickle.load(open(os.path.join(data_path, config['data']['train_loader']), 'rb'))
    texts = get_texts(train_loader, vocabulary)
    model = word2vec.Word2Vec(window=int(config['data']['window']), min_count=int(config['data']['min_count']), size=embed_dim)
    model.build_vocab(texts)
    model.train(texts, total_examples=model.corpus_count, epochs=model.epochs)
    weights = get_weights(model, vocabulary, embed_dim)
    pickle.dump(weights, open(os.path.join(data_path, config['data']['weights']), 'wb'))


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    # seed
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
