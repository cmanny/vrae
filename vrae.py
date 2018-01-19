import tensorflow
import os

from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

class VRAE(object):

    def __init__(self,
                batch_size=32,
                latent_size=20,
                num_layers=1,
                input_size=None,
                sequence_lengths=None,
                learning_rate=0.0001):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_size = input_size

        self.batch_input = tf.placeholder(
            tf.float32,
            shape=(batch_size, sequence_lengths, input_size),
            name="batch_input"
        )
        self._build_cg()

    def _build_cg(self):

        with tf.name_scope("encoder") as scope:
            cell = MultiRNNCell(
                [LSTMCell(self.input_size) for _ in range(self.num_layers)]
            )
            initial_state = cell.zero_state(self.batch_size, tf.float32)
            enc_outs, _ = tf.nn.dynamic_rnn(
                cell,
                inputs=self.batch_input,
                initial_state=initial_state
            )
            last_out = enc_outs[-1]
            W_ez = tf.get_variable(
                "ernn_to_lat_w",
                [self.input_size, self.latent_size]
            )
            b_ez = tf.get_variable("ernn_to_lat_b", [self.latent_size])
            self.z = tf.nn.xw_plus_b(last_out, W_ez, b_ez, name="Z")
            mean, variance = tf.nn.moments(self.z, axes=[0])
            self.lat_loss = tf.reduce_mean(
                tf.square(mean) + variance - tf.log(var) - 1
            )

        with tf.name_scope("decoder") as scope:
            W_zd = tf.get_variable(
                "lat_w_to_drnn",
                [self.latent_size, self.input_size]
            )
            b_zd = tf.get_variable("lat_b_to_drnn", [self.input_size])
            #decoder rnn to out


    def train_step(self, batch):
        pass

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass

if __name__ == "__main__":
    pass
