import tensorflow
import os
from collections import Counter
from scipy import signal
from scipy.io import wavfile
import wave

from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

class TIMITDataset(object):
    """/<CORPUS>/<USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>"""

    def __init__(self, base_path):
        self.base_path = base_path
        self.inner_path = "data/lisa/data/timit/raw/TIMIT"
        self.full_path = os.path.join(self.base_path, self.inner_path)

        self.dict_path = os.path.join(self.full_path, "DOC", "TIMITDIC.TXT")
        self.prompt_path = os.path.join(self.full_path, "DOC", "PROMPT.TXT")
        self.spkr_info_path = os.path.join(self.full_path, "DOC", "SPKRINFO.TXT")
        self.spkr_sent_path = os.path.join(self.full_path, "DOC", "SPKRSENT.TXT")

        self.regions = [
            "EMPTY"
            "New England",
            "Northern",
            "North Midland",
            "South Midland",
            "Southern",
            "New York City",
            "Western",
            "Army Brat"
        ]

        self._parse_dict()
        self._parse_spkr_info()
        self._parse_spkr_sent()

    def _word_and_phoneme_ids(self):
        self.word_to_id = {}
        self.phoneme_to_id = {}
        self.phoneme_counter = Counter()
        word_id = 0
        with open(self.dict_path, "r") as f:
            for line in f:
                if line[0] == ";":
                    continue
                word, phonemes = line.split("  ")
                phonemes = phonemes[1:-2].split()
                self.word_to_id[word] = word_id
                word_id += 1
                self.phoneme_counter += Counter(phonemes)
        for i, (phon, count) in enumerate(self.phoneme_counter.most_common()):
            self.phoneme_to_id[phon] = i

    def _parse_dict(self):
        self.word_id_to_phon_ids = {}
        self._word_and_phoneme_ids()
        with open(self.dict_path, "r") as f:
            for line in f:
                if line[0] == ";":
                    continue
                word, phonemes = line.split("  ")
                phonemes = phonemes[1:-2].split()
                self.word_id_to_phon_ids[self.word_to_id[word]] = [
                    self.phoneme_to_id[phon] for phon in phonemes
                ]

    def _parse_spkr_sent(self):
        self.spkr_sents = {}
        with open(self.spkr_sent_path, "r") as f:
            for line in f:
                if line[0] == ";":
                    continue
                line = [x.strip() for x in line.split()]
                self.spkr_sents[line[0]] = {
                    "SA" : line[1:3],
                    "SX" : line[3:9],
                    "SI" : line[9:12]
                }


    def _parse_spkr_info(self):
        self.speakers = {}
        with open(self.spkr_info_path, "r") as f:
            for line in f:
                if line[0] == ";":
                    continue
                s_info = [x.strip() for x in line.split()]
                if len(s_info) < 10:
                    s_info.append("")
                self.speakers[s_info[0]] = {
                    "sex" : s_info[1],
                    "dr"  : s_info[2],
                    "use" : s_info[3],
                    "rec_date" : s_info[4],
                    "birth_date" : s_info[5],
                    "height" : s_info[6],
                    "race" : s_info[7],
                    "ed" : s_info[8],
                    "comments" : s_info[9]
                }

    def stats(self):
        return {
            "num_speakers": len(self.speakers),
            "num_words": len(self.word_to_id),
            "num_phonemes": len(self.phoneme_to_id)
        }

    def _wav(self, file_path):
        #sph file fix
        print(file_path)
        sample_rate, samples = wave.open(file_path)

    def _wrd(self, file_path):
        wrd_list = []
        with open(file_path, "r") as f:
            for line in f:
                s, e, wrd = line.split()
                s, e = int(s), int(e)
                wrd_list.append((s, e, self.word_to_id[wrd]))
        return wrd_list

    def _phn(self, file_path):
        phn_list = []
        with open(file_path, "r") as f:
            for line in f:
                s, e, phn = line.split()
                s, e = int(s), int(e)
                wrd_list.append((s, e, self.phoneme_to_id[phn]))
        return phn_list

    def get_sentence_data(self, speaker, sid):
        spkr = self.speakers[speaker]
        part = "TRAIN" if spkr["use"] == "TRN" else "TEST"
        gend = spkr["sex"]
        dr = "DR" + spkr["dr"]
        folder = os.path.join(self.full_path, part, dr, gend + speaker)
        phn_file = os.path.join(folder, sid + ".PHN")
        wav_file = os.path.join(folder, sid + ".WAV")
        wrd_file = os.path.join(folder, sid + ".WRD")
        return self._wav(wav_file), self._wrd(wrd_file), self._phn(phn_file)






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
    timit = TIMITDataset('./TIMIT')
    print(timit.stats())
    print(timit.get_sentence_data("CJF0", "SA1"))
