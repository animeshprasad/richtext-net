"""
Generate data.
"""

import argparse
import json
import numpy as np
import os
import random
import re
import sys


def tokenize(text):
    """Tokenize a passage of text, i.e. return a list of words"""
    return text.split(' ')


class Loader(object):
    """Text data loader."""
    def __init__(self, data_path, vocab_path, batch_size, seq_length, n_pointers=2):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_length = seq_length

        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
            self.vocab_size = len(self.vocab)

        self.text = []
        self.embedding = None
        self.lengths = None
        self.labels = None
        self.n_batches = None
        self.x_batches, self.x_lengths, self.y_batches = None, None, None
        self.pointer = 0
        self.n_pointers = n_pointers

        print('Pre-processing data...')
        self.pre_process()
        self.create_batches()
        print('Pre-processed {} lines of data.'.format(self.labels.shape[0]))

    def get(self, something):
        return self.vocab.get(something, 1)

    def pre_process(self):
        """Pre-process data."""
        with open(self.data_path, 'r') as f:
            data = f.readlines()
        # each line in data file is formatted according to [label, text] (e.g. 2 She went home)
        self.all = [tokenize(sample.strip()) for sample in data]
        self.text = [' '.join(sample[self.n_pointers:]) for sample in self.all]
        self.labels = np.array([sample[:self.n_pointers] for sample in self.all])
        self.embedding = np.zeros((len(self.text), self.seq_length), dtype=int)
        self.lengths = np.zeros(len(self.text), dtype=int)
        for i, sample in enumerate(self.text):
            tokens = tokenize(self.text[i]) #No need punkt tokenizer
            self.lengths[i] = len(tokens)
            self.embedding[i] = list(map(self.get, tokens)) + [0] * (self.seq_length - len(tokens))


    def create_batches(self):
        """Split data into training batches."""
        self.n_batches = int(self.embedding.shape[0] / self.batch_size)
        # truncate training data so it is equally divisible into batches
        self.embedding = self.embedding[:self.n_batches * self.batch_size, :]
        self.lengths = self.lengths[:self.n_batches * self.batch_size]
        #print (self.labels.shape, self.x_batches.shape)
        self.labels = self.labels[:self.n_batches * self.batch_size, :]

        # split training data into equal sized batches
        self.x_batches = np.split(self.embedding, self.n_batches, 0)
        self.x_lengths = np.split(self.lengths, self.n_batches)
        self.y_batches = np.split(self.labels, self.n_batches, 0)

    def next_batch(self):
        """Return current batch, increment pointer by 1 (modulo n_batches)"""
        x, x_len, y = self.x_batches[self.pointer], self.x_lengths[self.pointer], self.y_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.n_batches
        return x, x_len, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory in which data is stored.')
    parser.add_argument('--save_dir', type=str, default='./models', help='Where to save checkpoint models.')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    args = parser.parse_args(sys.argv[1:])
    n_pointers = 2
    MAX_LENGTH = 500
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    training = Loader(os.path.join(args.data_dir, 'train.txt'), vocab_path, args.batch_size, MAX_LENGTH, n_pointers=n_pointers)
    validation = Loader(os.path.join(args.data_dir, 'validate.txt'), vocab_path, args.batch_size, MAX_LENGTH, n_pointers=n_pointers)