# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os

tf.logging.set_verbosity(tf.logging.ERROR)


class ARCADe:
    """ This class defines the model trained with the ARCADe algorithm.

    """

    def __init__(self, sess, args, seed, input_shape):
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.lr = args.lr
        self.meta_lr = args.meta_lr
        self.meta_epochs = args.meta_epochs
        self.K = args.K
        self.num_updates = args.num_updates

        self.update_mode = args.update_mode
        self.bn = args.bn
        self.arcade_h = args.arcade_h

        self.dataset = args.dataset
        self.sess = sess
        self.summary = False
        self.summary_dir = args.summary_dir

        if(self.summary_dir):
            self.summary = True
            self.summary_interval = 100
            summaries_list_metatrain = []
            summaries_list_val = []
            summaries_list_test_restore_val = []
        else:
            self.summary_dir = "no_summary"

        self.stop_grad = args.stop_grad
        self.n_queries = args.n_queries
        self.n_classes = 1
        self.input_shape = input_shape
        self.max_length = args.max_length

        # number of tasks to sample per meta-training iteration
        if(self.dataset == 'MIN' or self.dataset == 'CIFAR_FS'):
            self.n_sample_tasks = 4
        else:
            self.n_sample_tasks = 8

        self.padding = 'same'
        self.flatten = tf.keras.layers.Flatten()

        # build model
        self.layers = []
        if(args.filters == ""):
            self.filter_sizes = []
        else:
            self.filter_sizes = [int(i) for i in args.filters.split(' ')]
            self.kernel_sizes = [int(i) for i in args.kernel_sizes.split(' ')]
            if(len(self.filter_sizes) == 1):
                self.layers.append(
                    tf.keras.layers.Conv2D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0],
                        input_shape=(None,) + input_shape,
                        strides=1,
                        padding='same',
                        activation='relu',
                        name='conv_last'))

                if(self.bn):
                    self.layers.append(
                        tf.keras.layers.BatchNormalization(
                            name='bn_c_last'))

            else:
                self.layers.append(
                    tf.keras.layers.Conv2D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0],
                        input_shape=(None,) + input_shape,
                        strides=1,
                        padding='same',
                        activation='relu',
                        name='conv0'))
                if(self.bn):
                    self.layers.append(
                        tf.keras.layers.BatchNormalization(
                            name='bn_c0'))

            for i in range(1, len(self.filter_sizes)):
                if(i != len(self.filter_sizes) - 1):
                    self.layers.append(
                        tf.keras.layers.Conv2D(
                            filters=self.filter_sizes[i],
                            kernel_size=self.kernel_sizes[i],
                            strides=1,
                            padding='same',
                            activation='relu',
                            name='conv' + str(i)))
                    if(self.bn):
                        self.layers.append(
                            tf.keras.layers.BatchNormalization(
                                name='bn_c' + str(i)))

                else:
                    self.layers.append(
                        tf.keras.layers.Conv2D(
                            filters=self.filter_sizes[i],
                            kernel_size=self.kernel_sizes[i],
                            strides=1,
                            padding='same',
                            activation='relu',
                            name='conv_last'))
                    if(self.bn):
                        self.layers.append(
                            tf.keras.layers.BatchNormalization(
                                name='bn_c_last'))

        if(args.dense_layers == ""):
            self.dense_sizes = []
        else:
            self.dense_sizes = [int(i) for i in args.dense_layers.split(' ')]

        for i in range(0, len(self.dense_sizes)):
            self.layers.append(
                tf.keras.layers.Dense(
                    units=self.dense_sizes[i],
                    activation='relu',
                    name='dense' + str(i)))
            if(self.bn):
                self.layers.append(tf.keras.layers.BatchNormalization(
                    name='bn_d' + str(i + len(self.filter_sizes))))

        self.layers.append(
            tf.keras.layers.Dense(
                units=self.n_classes,
                name='dense_last'))

        # loss function
        self.loss_fct = tf.nn.sigmoid_cross_entropy_with_logits

        self.seq_length = tf.placeholder(
            tf.int32, (self.n_sample_tasks,), name='seq_length')
        self.X_train_a = [
            tf.placeholder(
                tf.float32,
                (None,
                 None) +
                input_shape,
                name='X_train_a' +
                str(i)) for i in range(
                self.n_sample_tasks)]
        self.Y_train_a = [
            tf.placeholder(
                tf.float32,
                (None,
                 None,
                 self.n_classes),
                name='Y_train_a' +
                str(i)) for i in range(
                self.n_sample_tasks)]
        self.X_train_b = [
            tf.placeholder(
                tf.float32,
                (None,
                 self.n_queries) +
                input_shape,
                name='X_train_b' +
                str(i)) for i in range(
                self.n_sample_tasks)]
        self.Y_train_b = [
            tf.placeholder(
                tf.float32,
                (None,
                 self.n_queries,
                 self.n_classes),
                name='Y_train_b' +
                str(i)) for i in range(
                self.n_sample_tasks)]

        self.X_finetune = tf.placeholder(
            tf.float32, (None, None,) + input_shape, name='X_finetune')
        self.Y_finetune = tf.placeholder(
            tf.float32, (None, None, self.n_classes), name='Y_finetune')
        self.X_test = tf.placeholder(
            tf.float32, (None, self.n_queries,) + input_shape, name='X_test')
        self.Y_test = tf.placeholder(
            tf.float32, (None, self.n_queries, self.n_classes), name='Y_test')

        self.construct_forward = tf.make_template(
            'construct_forward', self.feed_forward)

        self.finetune_output = self.construct_forward(
            self.X_finetune[0], training=True)
        self.finetune_loss = tf.reduce_mean(
            self.loss_fct(
                labels=self.Y_finetune[0],
                logits=self.finetune_output))

        self.m_vars = []
        self.lrs = []
        for layer_idx in range(0, len(self.layers)):
            self.m_vars.append(self.layers[layer_idx].weights[0])
            self.m_vars.append(self.layers[layer_idx].weights[1])

        if(self.arcade_h):
            for i in range(
                    len(self.m_vars[:(-2 - 2 * len(self.dense_sizes))])):
                self.lrs.append(self.lr)

            for i in range(
                    len(self.m_vars[:(-2 - 2 * len(self.dense_sizes))]), len(self.m_vars)):

                self.lrs.append(
                    tf.Variable(
                        tf.ones_like(
                            self.m_vars[i]) *
                        self.lr,
                        name='lr_' +
                        str(i)))

        else:

            for i in range(len(self.m_vars)):
                self.lrs.append(
                    tf.Variable(
                        tf.ones_like(
                            self.m_vars[i]) *
                        self.lr,
                        name='lr_' +
                        str(i)))

        self.l_loss, self.l_acc, self.r_loss, self.r_acc = self.finetune_test_op(
            self.X_finetune, self.Y_finetune, self.X_test, self.Y_test, training=True)

        if(self.bn):
            self.print_op = tf.print(
                'lr', [
                    (k, tf.reduce_mean(
                        self.lrs[k])) for k in range(
                        0, len(
                            self.lrs), 4)])
        else:
            self.print_op = tf.print(
                'lr', [
                    (k, tf.reduce_mean(
                        self.lrs[k])) for k in range(
                        0, len(
                            self.lrs), 2)])

        self.total_seqloss = self.compute_seqlosses(training=True)

        self.meta_optimizer_seq = tf.train.AdamOptimizer(self.meta_lr)

        self.meta_opt_compute_gradients_seq = self.meta_optimizer_seq.compute_gradients(
            self.total_seqloss)

        self.meta_update_op_seq = self.meta_optimizer_seq.apply_gradients(
            self.meta_opt_compute_gradients_seq)

        if(self.summary):
            summaries_list_metatrain.append(
                tf.summary.scalar('total_train_loss', self.total_seqloss))
            self.merged_metatrain = tf.summary.merge(
                summaries_list_metatrain)
            lrs_summaries = []
            for i in range(len(self.lrs)):
                lrs_summaries.append(
                    tf.summary.histogram(
                        'lr_' + str(i), self.lrs[i]))
            self.summary_lrs = tf.summary.merge(lrs_summaries)

        self.saver = tf.train.Saver()

        base_path = '/home/z003vdzb/Documents'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ubuntu/Projects'
        if (not (os.path.exists(base_path))):
            base_path = '/home/frikha_a/Projects'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ceesgniewyk/Projects'

        self.checkpoint_path = base_path + '/ARCADe/checkpoints_ARCADe/'
        if (not (os.path.exists(self.checkpoint_path))):
            os.mkdir(self.checkpoint_path)
        if (not (os.path.exists(os.path.join(self.checkpoint_path, self.summary_dir)))):
            os.mkdir(os.path.join(self.checkpoint_path, self.summary_dir))

    def compute_metrics(self, logits, labels, logits_are_predictions=False):
        """compute non-running performance metrics.

        Parameters
        ----------
        logits : tensor
        labels : tensor


        Returns
        -------
        acc : tensor
            accuracy.
        precision : tensor
            precision.
        recall : tensor
            recall.
        specificity : tensor
            specificity.
        f1_score : tensor
            F1 score.
        auc_pr : tensor
            AUC-PR.

        """
        if(logits_are_predictions):
            predictions = logits
        else:
            predictions = tf.cast(
                tf.greater(
                    tf.nn.sigmoid(logits),
                    0.5),
                tf.float32)
        TP = tf.count_nonzero(predictions * labels, dtype=tf.float32)
        TN = tf.count_nonzero((predictions - 1) *
                              (labels - 1), dtype=tf.float32)
        FP = tf.count_nonzero(predictions * (labels - 1), dtype=tf.float32)
        FN = tf.count_nonzero((predictions - 1) * labels, dtype=tf.float32)
        acc = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))

        precision = tf.cond(tf.math.equal((TP + FP), 0),
                            true_fn=lambda: 0.0, false_fn=lambda: TP / (TP + FP))
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        f1_score = tf.cond(
            tf.math.equal(
                (precision + recall),
                0),
            true_fn=lambda: 0.0,
            false_fn=lambda: 2 * precision * recall / (
                precision + recall))

        auc_pr = tf.metrics.auc(labels=labels, predictions=tf.nn.sigmoid(
            logits), curve='PR', summation_method='careful_interpolation')[1]

        return [acc, precision, recall, specificity, f1_score, auc_pr]

    def feed_forward(self, inp, training):
        """computes an output tensor by feeding the input through the network.

        Parameters
        ----------
        inp : tensor
            input tensor.
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        out : tensor
            output tensor.

        """
        if(len(self.input_shape) < 3 and len(self.filter_sizes) > 0):
            h = tf.expand_dims(inp, -1)
        else:
            h = inp

        n_layers_no_head = len(self.layers) - len(self.dense_sizes) - 1
        if(self.bn):
            n_layers_no_head = len(self.layers) - len(self.dense_sizes) * 2 - 1

        for i in range(n_layers_no_head):
            # print('i', i, self.layers[i].name, h.shape)
            if('conv' in self.layers[i].name):
                h = self.layers[i](h)
                h = tf.layers.max_pooling2d(
                    h, pool_size=2, strides=2, padding='same')
            elif('bn' in self.layers[i].name):
                h = self.layers[i](h, training=training)

            if(self.bn and 'bn_c_last' in self.layers[i].name):
                h = self.flatten(h)

            elif(not(self.bn) and 'conv_last' in self.layers[i].name):
                h = self.flatten(h)

        if(n_layers_no_head < 1):
            i = -1
        for j in range(i + 1, len(self.layers)):
            # print('j', j, self.layers[j].name, h.shape)

            h = self.layers[j](h)
        return h

    def get_first_updated_weights(self, loss):
        """computes the model parameters after the first adaptation/inner update of the first task in the sequence.

        Parameters
        ----------
        loss : tensor
            loss tensor.

        Returns
        -------
        new_weights : dict
            contains the parameters after applying the first adaptation
            update (the first theta prime).

        """

        # update only dense layers
        only_denses = False
        variables = self.m_vars

        grads = tf.gradients(loss, variables)

        if(self.stop_grad):
            grads = [tf.stop_gradient(grad) for grad in grads]

        self.lrs_dict = []

        if(not(self.bn)):
            w_keys = []
            b_keys = []

            for i in range(0, len(self.layers)):
                w_keys.append('w' + str(i + 1))
                b_keys.append('b' + str(i + 1))

            if(self.arcade_h):
                new_weights = dict(
                    zip(
                        w_keys,
                        [
                            variables[i] -
                            tf.keras.activations.relu(
                                self.lrs[i]) *
                            grads[i] if i == (
                                len(variables) -
                                2) else tf.identity(
                                variables[i]) for i in range(
                                0,
                                len(variables),
                                2)]))
                new_biases = dict(
                    zip(
                        b_keys,
                        [
                            variables[i] -
                            tf.keras.activations.relu(
                                self.lrs[i]) *
                            grads[i] if i == (
                                len(variables) -
                                1) else tf.identity(
                                variables[i]) for i in range(
                                1,
                                len(variables),
                                2)]))

            else:
                new_weights = dict(zip(w_keys, [variables[i] -
                                                tf.keras.activations.relu(self.lrs[i]) *
                                                grads[i] for i in range(0, len(variables), 2)]))
                new_biases = dict(zip(b_keys, [variables[i] -
                                               tf.keras.activations.relu(self.lrs[i]) *
                                               grads[i] for i in range(1, len(variables), 2)]))

            new_weights.update(new_biases)
            self.lrs_dict = dict(
                zip(w_keys, [self.lrs[i] for i in range(0, len(variables), 2)]))
            self.lrs_dict.update(
                dict(zip(b_keys, [self.lrs[i] for i in range(1, len(variables), 2)])))

        else:
            w_keys = []
            b_keys = []
            bn_gamma_keys = []
            bn_beta_keys = []

            for i in range(0, len(self.layers)):
                w_keys.append('w' + str(i + 1))
                b_keys.append('b' + str(i + 1))
                bn_gamma_keys.append('bn_gamma' + str(i + 1))
                bn_beta_keys.append('bn_beta' + str(i + 1))

            if(self.arcade_h):
                new_weights = dict(
                    zip(
                        w_keys,
                        [
                            variables[i] -
                            tf.keras.activations.relu(
                                self.lrs[i]) *
                            grads[i] if i == (
                                len(variables) -
                                2) else tf.identity(
                                variables[i]) for i in range(
                                0,
                                len(variables),
                                4)]))
                new_biases = dict(
                    zip(
                        b_keys,
                        [
                            variables[i] -
                            tf.keras.activations.relu(
                                self.lrs[i]) *
                            grads[i] if i == (
                                len(variables) -
                                2) else tf.identity(
                                variables[i]) for i in range(
                                1,
                                len(variables),
                                4)]))
                new_bn_gamma = dict(zip(bn_gamma_keys, [tf.identity(
                    variables[i]) for i in range(2, len(variables), 4)]))
                new_bn_beta = dict(zip(bn_beta_keys, [tf.identity(
                    variables[i]) for i in range(3, len(variables), 4)]))

            else:
                new_weights = dict(zip(w_keys, [variables[i] -
                                                tf.keras.activations.relu(self.lrs[i]) *
                                                grads[i] for i in range(0, len(variables), 4)]))
                new_biases = dict(zip(b_keys, [variables[i] -
                                               tf.keras.activations.relu(self.lrs[i]) *
                                               grads[i] for i in range(1, len(variables), 4)]))
                new_bn_gamma = dict(zip(bn_gamma_keys, [variables[i] - tf.keras.activations.relu(
                    self.lrs[i]) * grads[i] for i in range(2, len(variables), 4)]))
                new_bn_beta = dict(zip(bn_beta_keys, [variables[i] - tf.keras.activations.relu(
                    self.lrs[i]) * grads[i] for i in range(3, len(variables), 4)]))

            new_weights.update(new_biases)
            new_weights.update(new_bn_gamma)
            new_weights.update(new_bn_beta)

            self.lrs_dict = dict(
                zip(w_keys, [self.lrs[i] for i in range(0, len(variables), 4)]))
            self.lrs_dict.update(
                dict(zip(b_keys, [self.lrs[i] for i in range(1, len(variables), 4)])))
            self.lrs_dict.update(
                dict(zip(bn_gamma_keys, [self.lrs[i] for i in range(2, len(variables), 4)])))
            self.lrs_dict.update(
                dict(zip(bn_beta_keys, [self.lrs[i] for i in range(3, len(variables), 4)])))

        return new_weights

    def get_further_updated_weights(self, loss, old_weights):
        """computes the model parameters after one inner update.

        Parameters
        ----------
        loss : tensor
            loss tensor.
        old_weights : dict
            contains the parameters before applying the current inner update

        Returns
        -------
        new_weights : dict
            contains the parameters after applying the current inner update

        """

        old_weights_list = list(old_weights.values())
        grads = tf.gradients(loss, old_weights_list)

        if(self.stop_grad):
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(old_weights.keys(), grads))

        if(self.arcade_h):
            new_weights = dict(
                zip(
                    old_weights.keys(),
                    [
                        old_weights[key] -
                        tf.keras.activations.relu(
                            self.lrs_dict[key]) *
                        gradients[key] if str(
                            len(
                                self.filter_sizes) +
                            1) in key else tf.identity(
                            old_weights[key]) for key in old_weights.keys()]))

        else:
            new_weights = dict(zip(old_weights.keys(), [old_weights[key] - tf.keras.activations.relu(
                self.lrs_dict[key]) * gradients[key] for key in old_weights.keys()]))

        return new_weights

    def new_weights_construct_forward(self, inp, weights, training):
        """computes an output tensor by feeding the input tensor through
        the model parametrized with given weights.

        Parameters
        ----------
        inp : tensor
            input tensor.
        weights : dict
            contains the parameters after applying inner updates
            (one of the theta primes).
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        out : tensor
            output tensor.

        """
        epsilon = 0.001
        if(len(self.input_shape) < 3):
            h = tf.expand_dims(inp, -1)
        else:
            h = inp
        h = tf.cast(h, tf.float32)

        for i in range(0, len(self.filter_sizes)):
            h = tf.nn.conv2d(h,
                             filter=weights['w' + str(i + 1)],
                             strides=[1,
                                      1,
                                      1,
                                      1],
                             padding=self.padding.upper()) + weights['b' + str(i + 1)]
            h = tf.keras.activations.relu(h)
            h = tf.layers.max_pooling2d(
                h, pool_size=2, strides=2, padding=self.padding)
            if(self.bn):
                mean, var = tf.nn.moments(h, [0, 1, 2])
                h = ((h - mean) / tf.sqrt(var + epsilon)) * \
                    weights['bn_gamma' + str(i + 1)] + weights['bn_beta' + str(i + 1)]

        h = tf.layers.flatten(h)

        if(len(self.filter_sizes) == 0):
            i = 0

        if(len(self.dense_sizes) > 0):
            for j in range(i, len(self.filter_sizes) + len(self.dense_sizes)):
                h = tf.matmul(h, weights['w' + str(j + 1)]
                              ) + weights['b' + str(j + 1)]
                h = tf.keras.activations.relu(h)
                if(self.bn):
                    mean, var = tf.nn.moments(h, 0)
                    h = ((h - mean) / tf.sqrt(var + epsilon)) * \
                        weights['bn_gamma' + str(j + 1)] + weights['bn_beta' + str(j + 1)]

            out = tf.matmul(
                h, weights['w' + str(j + 2)]) + weights['b' + str(j + 2)]
        else:
            out = tf.matmul(
                h, weights['w' + str(i + 2)]) + weights['b' + str(i + 2)]
        return out

    def metatrain_seqtask(self, X_train_a, Y_train_a,
                          X_train_b, Y_train_b, training, seq_task_idx):
        """performs a meta-trainig iteration on a single meta-training task-sequence.

        Parameters
        ----------
        X_train_a : tensor
            contains features of the K datapoints sampled for the inner loop (adaptation) updates of each task in the sequence.
        Y_train_a : tensor
            contains labels of the K datapoints sampled for the inner loop (adaptation) updates of each task in the sequence.
        X_train_b : tensor
            contains features sampled for the outer loop updates of each task in the sequence.
        Y_train_b : tensor
            contains labels sampled for the outer loop updates of each task in the sequence.
        training : bool
            argument for Batch normalization layers.
        seq_task_idx: int
            index of the task in the task-sequence (needed to determine the sequence length)

        Returns
        -------
        total_loss : tensor
            loss of the meta-update for the given task-sequence.

        """

        train_a_output = self.construct_forward(
            X_train_a[0], training=training)
        train_a_loss = tf.reduce_mean(self.loss_fct(
            labels=Y_train_a[0],
            logits=train_a_output))
        new_weights = self.get_first_updated_weights(train_a_loss)

        current_weights = new_weights
        if('loss_from_random_previous_task' in self.update_mode):

            def cond_1(task_index, weights, metaloss):
                return tf.less(task_index, self.seq_length[seq_task_idx])

            loop_vars_1 = (tf.constant(0), current_weights, tf.constant(0.0))

            def body_1(task_index, weights, metaloss):

                loop_vars_2 = (task_index, tf.constant(0), weights)

                def cond_2(task_index, update_index, weights):
                    if(task_index == 0):
                        return tf.less(update_index, self.num_updates - 1)
                    else:
                        return tf.less(update_index, self.num_updates)

                def body_2(task_index, update_index, weights):
                    train_a_output = self.new_weights_construct_forward(
                        X_train_a[task_index], weights, training=training)
                    train_a_loss = tf.reduce_mean(self.loss_fct(
                        labels=Y_train_a[task_index],
                        logits=train_a_output))
                    new_weights_loop = self.get_further_updated_weights(
                        train_a_loss, weights)

                    return task_index, tf.add(
                        update_index, 1), new_weights_loop

                z, i, new_weights_aft_task = tf.while_loop(
                    cond_2, body_2, loop_vars_2, swap_memory=True)

                train_b_output = self.new_weights_construct_forward(
                    X_train_b[task_index], new_weights_aft_task, training=training)

                train_b_loss = tf.reduce_mean(self.loss_fct(
                    labels=Y_train_b[task_index],
                    logits=train_b_output))

                metaloss = tf.add(metaloss, train_b_loss)

                def evaluate_on_random_prev_task(
                        task_index,
                        X_train_b,
                        Y_train_b,
                        new_weights_aft_task,
                        metaloss,
                        training):
                    k = tf.random.uniform(
                        shape=(
                            1,
                        ),
                        minval=0,
                        maxval=task_index,
                        dtype=tf.dtypes.int32)
                    train_b_output_prev_task = self.new_weights_construct_forward(tf.squeeze(
                        tf.gather(X_train_b, k), axis=0), new_weights_aft_task, training=training)

                    train_b_loss_prev_task = tf.reduce_mean(self.loss_fct(
                        labels=tf.squeeze(tf.gather(Y_train_b, k), axis=0),
                        logits=train_b_output_prev_task))
                    return tf.add(metaloss, train_b_loss_prev_task)

                metaloss = tf.cond(
                    task_index > tf.constant(0),
                    lambda: evaluate_on_random_prev_task(
                        task_index,
                        X_train_b,
                        Y_train_b,
                        new_weights_aft_task,
                        metaloss,
                        training),
                    lambda: tf.identity(metaloss))

                return tf.add(task_index, 1), new_weights_aft_task, metaloss

            y, new_weights_aft_all_tasks, metaloss_tmp = tf.while_loop(
                cond_1, body_1, loop_vars_1, swap_memory=True)

            def cond_3(task_index, metaloss):
                return tf.less(task_index, self.seq_length[seq_task_idx])

            loop_vars_3 = (tf.constant(0), metaloss_tmp)

            def body_3(task_index, metaloss):
                train_b_output = self.new_weights_construct_forward(
                    X_train_b[task_index], new_weights_aft_all_tasks, training=training)

                train_b_loss = tf.reduce_mean(self.loss_fct(
                    labels=Y_train_b[task_index],
                    logits=train_b_output))
                return tf.add(task_index, 1), tf.add(metaloss, train_b_loss)

            j, total_loss = tf.while_loop(
                cond_3, body_3, loop_vars_3, swap_memory=True)

            return total_loss

        else:
            print('loss not implemented')
            assert(0)

    def compute_seqlosses(self, training):
        """computes the total loss over all task-sequences (loss for the meta-update).

        Parameters
        ----------
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        total_loss : tensor
            sum of the meta-losses computed for each meta-training task-sequence.

        """

        total_loss = 0
        for index in range(self.n_sample_tasks):
            total_loss += self.metatrain_seqtask(
                self.X_train_a[index],
                self.Y_train_a[index],
                self.X_train_b[index],
                self.Y_train_b[index],
                training,
                index)

        return total_loss

    def metatrain_op(self, epoch, X_train_a, Y_train_a, X_train_b, Y_train_b):
        """performs one meta-training iteration.

        Parameters
        ----------
        X_train_a : tensor
            contains features of the K datapoints sampled for the inner loop (adaptation) updates of each meta-training task-sequence in the batch.
        Y_train_a : tensor
            contains labels of the K datapoints sampled for the inner loop (adaptation) updates of each meta-training task-sequence in the batch.
        X_train_b : tensor
            contains features sampled for the outer loop updates of each meta-training task-sequence in the batch.
        Y_train_b : tensor
            contains labels sampled for the outer loop updates of each meta-training task-sequence in the batch.

        Returns
        -------
        metatrain_loss : float
            sum of the meta-losses computed on the sampled meta-training task-sequences (the loss used for the meta-update).
        train_summaries : list
            training summaries.

        """

        feed_dict_train = {
            self.seq_length: np.array([len(x) for x in X_train_a]),
        }
        feed_dict_train.update(
            {ph: X_train_a_data for ph, X_train_a_data in zip(self.X_train_a, X_train_a)})
        feed_dict_train.update(
            {ph: Y_train_a_data for ph, Y_train_a_data in zip(self.Y_train_a, Y_train_a)})
        feed_dict_train.update(
            {ph: X_train_b_data for ph, X_train_b_data in zip(self.X_train_b, X_train_b)})
        feed_dict_train.update(
            {ph: Y_train_b_data for ph, Y_train_b_data in zip(self.Y_train_b, Y_train_b)})

        if(epoch % 100 == 0):
            self.sess.run(self.print_op)
        if(seq):
            metatrain_loss, _ = self.sess.run(
                [self.total_seqloss, self.meta_update_op_seq], feed_dict_train)
        else:
            metatrain_loss, _ = self.sess.run(
                [self.total_loss, self.meta_update_op], feed_dict_train)

        train_summaries = None

        return metatrain_loss, train_summaries

    def finetune_test_op(
            self,
            X_finetune,
            Y_finetune,
            X_test,
            Y_test,
            training):
        """performs finetuning (adaptation) and testing on a single meta-validation or meta-testing task-sequence.

        Parameters
        ----------
        X_finetune : tensor
            contains features of the K datapoints sampled for the inner loop (adaptation) updates of each task in the sequence.
        Y_finetune : tensor
            contains labels of the K datapoints sampled for the inner loop (adaptation) updates of each task in the sequence.
        X_test : tensor
            contains features sampled for the outer loop updates of each task in the sequence.
        Y_test : tensor
            contains labels sampled for the outer loop updates of each task in the sequence.
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        l_loss_sum : tensor
            sum of the learning losses on the test sets computed for the given the given task-sequence.
            The learning loss for a particular task is computed immediately after adaptation.
        l_acc_sum : tensor
            sum of the learning accuracies on the test sets computed for the given the given task-sequence.
            The learning accuracy for a particular task is computed immediately after adaptation.
        r_loss_sum : tensor
            sum of the retained losses on the test sets computed for the given the given task-sequence.
            The retained loss for a particular task is computed after learning all the tasks in the sequence.
        r_acc_sum : tensor
            sum of the retained accuracies on the test sets computed for the given the given task-sequence.
            The retained accuracy for a particular task is computed after learning all the tasks in the sequence.

        """

        finetune_output = self.construct_forward(
            X_finetune[0], training=training)
        finetune_loss = tf.reduce_mean(self.loss_fct(
            labels=Y_finetune[0],
            logits=finetune_output))
        new_weights = self.get_first_updated_weights(finetune_loss)

        current_weights = new_weights

        def cond_1(task_index, weights, l_loss, acc):
            return tf.less(task_index, self.seq_length[0])

        loop_vars_1 = (
            tf.constant(0),
            current_weights,
            tf.constant(0.0),
            tf.constant(0.0))

        def body_1(task_index, weights, l_loss, acc):

            loop_vars_2 = (task_index, tf.constant(0), weights)

            def cond_2(task_index, update_index, weights):
                if(task_index == 0):
                    return tf.less(update_index, self.num_updates - 1)
                else:
                    return tf.less(update_index, self.num_updates)

            def body_2(task_index, update_index, weights):
                finetune_output = self.new_weights_construct_forward(
                    X_finetune[task_index], weights, training=training)
                finetune_loss = tf.reduce_mean(self.loss_fct(
                    labels=Y_finetune[task_index],
                    logits=finetune_output))
                new_weights_loop = self.get_further_updated_weights(
                    finetune_loss, weights)

                return task_index, tf.add(update_index, 1), new_weights_loop

            z, i, new_weights_aft_task = tf.while_loop(
                cond_2, body_2, loop_vars_2, swap_memory=True)

            test_output = self.new_weights_construct_forward(
                X_test[task_index], new_weights_aft_task, training=training)

            test_loss = tf.reduce_mean(self.loss_fct(
                labels=Y_test[task_index],
                logits=test_output))

            predictions = tf.cast(
                tf.greater(
                    tf.nn.sigmoid(test_output),
                    0.5),
                tf.float32)

            acc_task = tf.reduce_mean(
                tf.to_float(
                    tf.equal(
                        predictions,
                        Y_test[task_index])))
            acc = tf.add(acc, acc_task)

            l_loss = tf.add(l_loss, test_loss)

            return tf.add(task_index, 1), new_weights_aft_task, l_loss, acc

        y, new_weights_aft_all_tasks, l_loss_sum, l_acc_sum = tf.while_loop(
            cond_1, body_1, loop_vars_1, swap_memory=True)

        def cond_3(task_index, r_loss, acc):
            return tf.less(task_index, self.seq_length[0])

        loop_vars_3 = (tf.constant(0), tf.constant(0.0), tf.constant(0.0))

        def body_3(task_index, r_loss, acc):
            test_output = self.new_weights_construct_forward(
                X_test[task_index], new_weights_aft_all_tasks, training=training)

            test_loss = tf.reduce_mean(self.loss_fct(
                labels=Y_test[task_index],
                logits=test_output))

            predictions = tf.cast(
                tf.greater(
                    tf.nn.sigmoid(test_output),
                    0.5),
                tf.float32)

            acc_task = tf.reduce_mean(
                tf.to_float(
                    tf.equal(
                        predictions,
                        Y_test[task_index])))

            return tf.add(
                task_index, 1), tf.add(
                r_loss, test_loss), tf.add(
                acc, acc_task)

        j, r_loss_sum, r_acc_sum = tf.while_loop(
            cond_3, body_3, loop_vars_3, swap_memory=True)

        return l_loss_sum, l_acc_sum, r_loss_sum, r_acc_sum

    def val_op_seq(self, K_X_val, K_Y_val, val_test_X, val_test_Y):
        """performs one validation episode.

        Parameters
        ----------
        K_X_val : array
            contains features of the K datapoints sampled for adaptation to the validation task-sequence.
        K_Y_val : array
            contains labels of the K datapoints sampled for adaptation to the validation task-sequence.
        val_test_X : array
            contains features of the test set(s) of the validation task-sequence.
        val_test_Y : array
            contains labels of the test set(s) of the validation task-sequence.

        Returns
        -------
        val_summaries : list
            validation summaries.
        learning_metrics : list
            learning loss and accuracy for the task-sequence.
        retained_metrics : float
            retained loss and accuracy for the task-sequence.
        bti_metrics : float
            backward transfer and interference (BTI) metrics, i.e. difference between retained and learning metrics.

        """

        # save current network parameters (including bn stats)
        old_vars = []
        for layer_idx in range(0, len(self.layers)):
            layer_weights = self.layers[layer_idx].get_weights()
            old_vars.append(layer_weights)

        self.sess.run(tf.local_variables_initializer())

        seq_length = len(K_X_val)
        feed_dict = {
            self.seq_length: np.array([seq_length for _ in range(self.n_sample_tasks)]),
            self.X_finetune: K_X_val,
            self.Y_finetune: K_Y_val,
            self.X_test: val_test_X,
            self.Y_test: val_test_Y,
        }

        l_loss, l_acc, r_loss, r_acc = self.sess.run(
            [self.l_loss, self.l_acc, self.r_loss, self.r_acc], feed_dict=feed_dict)

        l_loss, l_acc, r_loss, r_acc = l_loss / seq_length, l_acc / \
            seq_length, r_loss / seq_length, r_acc / seq_length
        val_summaries = None

        learning_metrics = [l_loss, l_acc]
        retained_metrics = [r_loss, r_acc]
        bti_metrics = [r_loss - l_loss, r_acc - l_acc]

        return val_summaries, learning_metrics, retained_metrics, bti_metrics
