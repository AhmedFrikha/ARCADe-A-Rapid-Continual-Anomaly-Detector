# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import random
import pickle
import json

from miniimagenet_tasks import create_miniimagenet_data_split
from omniglot_tasks import get_omniglot_allcharacters_data_split
from cifarfs_tasks import create_cifarfs_data_split
from task import TaskAsSequenceOfTasks
from arcade_class import ARCADe


tf.logging.set_verbosity(tf.logging.ERROR)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def main(args):

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    cir = args.cir_inner_loop
    K = args.K
    min_length = min_length_val = args.min_length
    max_length = max_length_val = args.max_length

    if(cir == 0.5):
        args.train_occ = False
    elif(cir == 1.0):
        args.train_occ = True
    else:
        print('not implemented - cir must be either 0.5 or 1.0')
        assert(0)

    args.test_occ = True

    # load the data
    base_path = "/home/z003vdzb/Documents/"
    if not (os.path.exists(base_path)):
        base_path = "/home/ubuntu/Projects/"
    if not (os.path.exists(base_path)):
        base_path = "/home/frikha_a/Projects/"

    basefolder = base_path + "MAML/raw_data/"

    if('MIN' in args.dataset):
        trX, trY, valX, valY, teX, teY = create_miniimagenet_data_split(
            basefolder + "miniImageNet_data/miniimagenet.pkl")
    elif('CIFAR_FS' in args.dataset):
        trX, trY, valX, valY, teX, teY = create_cifarfs_data_split(
            base_path + "MAML/cifar_fc100/data/CIFAR_FS/CIFAR_FS_train.pickle",
            base_path + "MAML/cifar_fc100/data/CIFAR_FS/CIFAR_FS_val.pickle",
            base_path + "MAML/cifar_fc100/data/CIFAR_FS/CIFAR_FS_test.pickle")
    else:
        trX, trY, valX, valY, teX, teY = get_omniglot_allcharacters_data_split(
            basefolder + "omniglot/omniglot.pkl")

    metatrain_task = TaskAsSequenceOfTasks(
        trX,
        trY,
        min_length,
        max_length,
        num_training_samples_per_class=args.K,
        num_test_samples_per_class=int(
            args.n_queries / 2),
        train_occ=args.train_occ)

    metaval_task = TaskAsSequenceOfTasks(
        valX,
        valY,
        min_length_val,
        max_length_val,
        num_training_samples_per_class=args.K,
        num_test_samples_per_class=int(
            args.n_queries / 2),
        train_occ=args.test_occ)

    # build model
    input_shape = metatrain_task.current_task_sequence[0].get_train_set()[
        0][0].shape
    sess = tf.InteractiveSession()
    model = ARCADe(sess, args, seed, input_shape)

    # summary logs
    summary = False
    if(args.summary_dir):
        summary = True

    if(summary):
        loddir_path = './summaries_ARCADe'
        if (not (os.path.exists(loddir_path))):
            os.mkdir(loddir_path)
        if (not (os.path.exists(os.path.join(loddir_path, model.summary_dir)))):
            os.mkdir(os.path.join(loddir_path, model.summary_dir))
        train_writer = tf.summary.FileWriter(
            os.path.join(loddir_path, model.summary_dir) + '/train')
        val_writer = tf.summary.FileWriter(
            os.path.join(loddir_path, model.summary_dir) + '/val')
        val_tags = ['loss', 'acc', 'prec', 'rec', 'spec', 'f1', 'auc']

    # initialize model
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # intialization of training hyperparameters
    init_epoch = 0
    n_test_tasks = 100
    n_val_tasks_sampled = 10
    val_test_loss = 0
    min_val_epoch = -1
    min_val_test_loss = 10000
    max_val_acc = 0
    min_metatrain_epoch = -1
    min_metatrain_loss = 10000
    val_interval = 100

    for epoch in range(init_epoch, args.meta_epochs + init_epoch):

        if((epoch % val_interval == 0) or (epoch == args.meta_epochs + init_epoch - 1)):
            # perform a validation episode using meta-validation task-sequences
            val_metrics_list = []
            for _ in range(n_val_tasks_sampled):
                metaval_task.reset()
                X_val_a, Y_val_a, X_val_b, Y_val_b = [], [], [], []
                for task_i in range(len(metaval_task.current_task_sequence)):
                    task = metaval_task.current_task_sequence[task_i]
                    X_val_a.append(task.get_train_set()[0])
                    Y_val_a.append(np.expand_dims(task.get_train_set()[1], -1))
                    X_val_b.append(task.get_test_set()[0])
                    Y_val_b.append(np.expand_dims(task.get_test_set()[1], -1))

                val_summaries, learning, retained, bti = model.val_op_seq(
                    X_val_a, Y_val_a, X_val_b, Y_val_b)
                val_metrics_list.append([learning, retained, bti])

            avg_val_metrics = np.mean(val_metrics_list, axis=0)

            # save restore based on retained accuracy
            if(avg_val_metrics[1][1] > max_val_acc):
                model.saver.save(
                    model.sess,
                    model.checkpoint_path +
                    model.summary_dir +
                    "_r_testacc_seq/model.ckpt")
                min_val_test_loss = avg_val_metrics[1][0]
                max_val_acc = avg_val_metrics[1][1]
                min_val_epoch = epoch
                print(' ***** model saved *****')

            print('++ epoch: ', epoch, '-L-'
                  'loss:', avg_val_metrics[0][0],
                  ' acc :', avg_val_metrics[0][1],
                  )
            print('++ epoch: ', epoch, '-R-'
                  'loss:', avg_val_metrics[1][0],
                  ' acc :', avg_val_metrics[1][1],
                  )
            print('++ epoch: ', epoch, '-B-'
                  'loss:', avg_val_metrics[2][0],
                  ' acc :', avg_val_metrics[2][1],
                  )

            if(summary):
                val_summaries = []
                val_summaries.append(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(
                                tag='l_accuracy',
                                simple_value=avg_val_metrics[0][1]),
                        ]))
                val_summaries.append(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(
                                tag='r_accuracy',
                                simple_value=avg_val_metrics[1][1]),
                        ]))
                val_summaries.append(
                    tf.Summary(
                        value=[
                            tf.Summary.Value(
                                tag='bti',
                                simple_value=avg_val_metrics[2][1]),
                        ]))

            for smr in val_summaries:
                val_writer.add_summary(smr, epoch)
            if(hasattr(model, 'summary_lrs')):
                lrs_smr = model.sess.run(model.summary_lrs)
                val_writer.add_summary(lrs_smr, epoch)
            val_writer.flush()

        # perform a meta-training iteration using meta-training task-sequences
        X_train_a, Y_train_a, X_train_b, Y_train_b = [], [], [], []
        for _ in range(model.n_sample_tasks):
            metatrain_task.reset()
            X_train_a_per_task, Y_train_a_per_task, X_train_b_per_task, Y_train_b_per_task = [], [], [], []

            for task_i in range(len(metatrain_task.current_task_sequence)):
                task = metatrain_task.current_task_sequence[task_i]
                X_train_a_per_task.append(task.get_train_set()[0])
                Y_train_a_per_task.append(
                    np.expand_dims(
                        task.get_train_set()[1], -1))
                X_train_b_per_task.append(task.get_test_set()[0])
                Y_train_b_per_task.append(
                    np.expand_dims(
                        task.get_test_set()[1], -1))

            X_train_a.append(X_train_a_per_task)
            Y_train_a.append(Y_train_a_per_task)
            X_train_b.append(X_train_b_per_task)
            Y_train_b.append(Y_train_b_per_task)

        X_train_a = np.array(X_train_a)
        Y_train_a = np.array(Y_train_a)
        X_train_b = np.array(X_train_b)
        Y_train_b = np.array(Y_train_b)

        metatrain_loss, train_summaries = model.metatrain_op(
            epoch, X_train_a, Y_train_a, X_train_b, Y_train_b)

        if(metatrain_loss < min_metatrain_loss):
            min_metatrain_loss = metatrain_loss
            print('epoch :', epoch, 'min_metatrain_loss:', min_metatrain_loss)

    if(summary):
        train_writer.close()
        val_writer.close()

    # restore best model params
    model.saver.restore(
        model.sess,
        model.checkpoint_path +
        model.summary_dir +
        "_r_testacc_seq/model.ckpt")

    if('MIN' in args.dataset or 'CIFAR_FS' in args.dataset):
        test_task_seq_lengths = [1, 2, 3, 4, 5]
    else:
        test_task_seq_lengths = [1, 2, 10, 20, 40, 75, 100]

    # meta-testing on task-sequences with different lengths
    for test_seq_length in test_task_seq_lengths:
        metatest_task = TaskAsSequenceOfTasks(
            teX,
            teY,
            test_seq_length,
            test_seq_length,
            num_training_samples_per_class=args.K,
            num_test_samples_per_class=int(
                args.n_queries / 2),
            train_occ=args.test_occ)

        print('test_seq_length', test_seq_length)

        test_metrics_list = []
        for i in range(n_test_tasks):
            metatest_task.reset()
            X_test_a, Y_test_a, X_test_b, Y_test_b = [], [], [], []
            for task_i in range(len(metatest_task.current_task_sequence)):
                task = metatest_task.current_task_sequence[task_i]
                X_test_a.append(task.get_train_set()[0])
                Y_test_a.append(np.expand_dims(task.get_train_set()[1], -1))
                X_test_b.append(task.get_test_set()[0])
                Y_test_b.append(np.expand_dims(task.get_test_set()[1], -1))

            test_summaries, learning, retained, bti = model.val_op_seq(
                X_test_a, Y_test_a, X_test_b, Y_test_b)
            test_metrics_list.append([learning, retained, bti])

        avg_test_metrics = np.mean(test_metrics_list, axis=0)

        print('Average -L-'
              'loss:', avg_test_metrics[0][0],
              ' acc :', avg_test_metrics[0][1],
              )
        print('++ -R-'
              'loss:', avg_test_metrics[1][0],
              ' acc :', avg_test_metrics[1][1],
              )
        print('++ -B-'
              'loss:', avg_test_metrics[2][0],
              ' acc :', avg_test_metrics[2][1],
              )

    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ARCADe: A Rapid Continual Anomaly Detector')
    parser.add_argument('-config_file',
                        type=str,
                        default="None")

    args = parser.parse_args()

    args_dict = vars(args)
    if args.config_file is not "None":
        args_dict = extract_args_from_json(args.config_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False

    args = Bunch(args_dict)

    assert(args.num_updates > 0), ("at least one update is needed")
    main(args)
