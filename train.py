"""
Train PointerNet model.
"""

import argparse
import os
import tensorflow as tf
import sys
from data import Loader
from model import PointerNet
from tqdm import tqdm
import numpy as np


def train(args):
    # load data
    n_pointers = 2
    MAX_LENGTH = 30

    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    training = Loader(os.path.join(args.data_dir, 'train.txt'), vocab_path, args.batch_size, MAX_LENGTH, n_pointers=n_pointers)
    validation = Loader(os.path.join(args.data_dir, 'validate.txt'), vocab_path, args.batch_size, MAX_LENGTH, n_pointers=n_pointers)
    testing = Loader(os.path.join(args.data_dir, 'test.txt'), vocab_path, args.batch_size, MAX_LENGTH,
                        n_pointers=n_pointers)


    # create TensorFlow graph
    ptr_net = PointerNet(batch_size=args.batch_size, learning_rate=args.learning_rate, n_pointers=n_pointers)
    saver = tf.train.Saver()
    best_val_acc = 0

    # record training loss & accuracy
    train_losses = []
    train_accuracies = []

    # initialize graph
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for ep in tqdm(range(args.n_epochs)):
            tr_loss, tr_acc = 0, 0
            for itr in range(training.n_batches):
                x_batch, x_lengths, y_batch = training.next_batch()
                train_dict = {ptr_net.encoder_inputs: x_batch,
                              ptr_net.input_lengths: x_lengths,
                              ptr_net.pointer_labels: y_batch}
                loss, acc, _ = sess.run([ptr_net.loss, ptr_net.exact_match, ptr_net.train_step], feed_dict=train_dict)
                tr_loss += loss
                tr_acc += acc

            train_losses.append(tr_loss / training.n_batches)
            train_accuracies.append(tr_acc / training.n_batches)

            # check validation accuracy every 10 epochs
            if ep % 10 == 0:
                val_acc = 0
                for itr in range(validation.n_batches):
                    x_batch, x_lengths, y_batch = validation.next_batch()
                    val_dict = {ptr_net.encoder_inputs: x_batch,
                                ptr_net.input_lengths: x_lengths,
                                ptr_net.pointer_labels: y_batch}
                    val_acc += sess.run(ptr_net.exact_match, feed_dict=val_dict)
                val_acc = val_acc / validation.n_batches

                print('epoch {:3d}, loss={:.2f}'.format(ep, tr_loss / training.n_batches))
                print('Train EM: {:.2f}, Validation EM: {:.2f}'.format(tr_acc / training.n_batches, val_acc))

                # save model
                if val_acc >= best_val_acc:
                    print('Validation accuracy increased. Saving model.')
                    saver.save(sess, os.path.join(args.save_dir, 'ptr_net.ckpt'))
                    best_val_acc = val_acc
                else:
                    print('Validation accuracy decreased. Restoring model.')
                    saver.restore(sess, os.path.join(args.save_dir, 'ptr_net.ckpt'))

        print('Training complete.')
        print('Best Validation EM: {:.2f}'.format(best_val_acc))

        output_ptr0 = []
        output_ptr1 = []
        labels_ptr0 = []
        labels_ptr1 = []
        for itr in range(testing.n_batches):
            x_batch, x_lengths, y_batch = testing.next_batch()
            val_dict = {ptr_net.encoder_inputs: x_batch,
                        ptr_net.input_lengths: x_lengths,
                        ptr_net.pointer_labels: y_batch}
            outs_ptr0, outs_ptr1, labels0, labels1 = sess.run([ptr_net.outs_ptr0, ptr_net.outs_ptr1, ptr_net.labels0, ptr_net.labels1], feed_dict=val_dict)

            print (np.array(labels0).shape, np.array(outs_ptr0).shape)
            output_ptr0.append(np.squeeze(outs_ptr0))
            output_ptr1.append(np.squeeze(outs_ptr1))
            labels_ptr0.append(np.squeeze(labels0))
            labels_ptr1.append(np.squeeze(labels1))

        output_ptr0 = np.vstack(output_ptr0)
        output_ptr1 = np.vstack(output_ptr1)
        labels_ptr0 = np.vstack(labels_ptr0).flatten()
        labels_ptr1 = np.vstack(labels_ptr1).flatten()
        print (output_ptr0.shape, labels_ptr0.shape)

        index = np.argmax(output_ptr0, -1)

    with open("output/ptr_test.txt", 'w') as outfile:
        for i in range(len(output_ptr0)):
            # print (output_ptr0[i])
            # print (labels_ptr0[i])
            # print (output_ptr1[i])
            # print (labels_ptr1[i])
            # print ('---')
            start_longest_seq = index[i]
            end_longest_seq = np.argmax(output_ptr1[i][index[i]:]) + index[i]
            try:
                outfile.write(str(start_longest_seq) + ' ' + str(end_longest_seq) + ' -1 -1 \n')
            except:
                outfile.write('0' + ' ' + '0' + ' -1 -1 \n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory in which data is stored.')
    parser.add_argument('--save_dir', type=str, default='./models', help='Where to save checkpoint models.')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to run.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    args = parser.parse_args(sys.argv[1:])
    train(args)
