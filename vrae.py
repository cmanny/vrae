import tensorflow as tf
import os

from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple
from tf_ops import linear

# Adapted from https://github.com/RobRomijnders/AE_ts/blob/master/AE_ts_model.py
# and https://github.com/hardmaru/diff-vae-tensorflow/blob/master/model.py



class VRAE(object):

    def __init__(self,
                batch_size=32,
                latent_size=20,
                num_layers=1,
                input_size=1024,
                hidden_size=128,
                sequence_lengths=None,
                learning_rate=0.001):
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.batch_input = tf.placeholder(
            tf.float32,
            shape=(None, None, input_size),
            name="batch_input"
        )
        seq_output, z = self._build_model(self.batch_input)
        self._build_loss_optimizer(seq_output, z)

        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        self.summary_op = tf.summary.merge_all()

        self.sess.run(init)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)


    def _length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def _recognizer(self, x):
        with tf.name_scope("recognizer") as scope:
            hidden = tf.nn.softplus(linear(x, self.hidden_size, 'r_x_to_hidden'))
            mean = linear(hidden, self.latent_size, 'r_hidden_to_mean')
            log_var = linear(hidden, self.latent_size, 'r_hidden_to_var')
            return mean, log_var


    def _generator(self, z):
        with tf.name_scope("generator") as scope:
            hidden = tf.nn.softplus(linear(z, self.hidden_size, 'g_z_to_hidden'))
            self.x_reconstructed = linear(hidden, self.input_size, 'g_hidden_to_x')
            return self.x_reconstructed

    def _rnn_encoder(self, sequence):
        with tf.variable_scope("sequence_encoder"):
            in_cell = MultiRNNCell(
                [LSTMCell(self.input_size) for _ in range(self.num_layers)]
            )
            initial_state = in_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.length = self._length(sequence)
            enc_outs, _ = tf.nn.dynamic_rnn(
                in_cell,
                inputs=sequence,
                initial_state=initial_state,
                sequence_length=self.length
            )
            return enc_outs[tf.squeeze(self.length) - 1]

    def _rnn_decoder(self, state):
        with tf.variable_scope("sequence_decoder"):
            out_cell = MultiRNNCell(
                [LSTMCell(self.input_size) for _ in range(self.num_layers)]
            )
            initial_state = (LSTMStateTuple(state, state),) * self.num_layers
            inputs = tf.zeros(
                (self.batch_size, tf.reduce_max(self.length), self.input_size)
            )
            dec_outs, _ = tf.nn.dynamic_rnn(
                out_cell,
                inputs=inputs,
                initial_state=initial_state,
                sequence_length=self.length
            )
            return dec_outs


    def _build_model(self, sequence_input):
        sequence_vector = self._rnn_encoder(sequence_input)
        self.in_mean, self.in_log_var = self._recognizer(sequence_vector)
        epsilon = tf.random_normal(
            [self.batch_size, self.latent_size],
            0, 1, dtype=tf.float32
        )
        self.z = self.in_mean + epsilon * tf.sqrt(tf.exp(self.in_log_var))
        decoded_state = self._generator(self.z)
        self.sequence_output = self._rnn_decoder(decoded_state)
        return self.sequence_output, self.z

    def _build_loss_optimizer(self, sequence_output, z):
        with tf.name_scope("optimizer"):
            out_flatten = tf.reshape(sequence_output, [self.batch_size, -1])
            in_flatten = tf.reshape(self.batch_input, [self.batch_size, -1])

            #L2 distance
            dist = out_flatten - in_flatten
            d2 = dist * dist * 2
            self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(d2, 1))
            self.kl_divergence = 0
            self.loss = self.reconstr_loss + self.kl_divergence
            self.t_vars = tf.trainable_variables()
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate
            ).minimize(self.loss, var_list=self.t_vars)


    def train_batch(self, batch):
        loss, kl, rloss = self.sess.run(
            self.loss,
            self.kl_divergence,
            self.reconstr_loss,
            feed_dict={
                self.batch_input: batch
            }
        )
        return loss, kl, rloss

    def save(self, file_path='ckpt/default', epoch=1):
        summary_writer = tf.summary.FileWriter(file_path, self.sess.graph)
        summary_str = self.sess.run(self.summary_op)
        summary_writer.add_summary(summary_str, 0)
        self.saver.save(self.sess, file_path, global_step=epoch)

    def load(self, file_path):
        ckpt = tf.train.get_checkpoint_state(file_path)
        self.saver.restore(self.sess, file_path+'/'+ckpt.model_checkpoint_path)

if __name__ == "__main__":
    vrae = VRAE()
    vrae.save()
