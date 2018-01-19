from collections import Counter
from scipy import signal
from scipy.io import wavfile
from sphfile import SPHFile
import os
import pprint
import pickle

class TIMITDataset(object):
    """/<CORPUS>/<USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>"""

    def __init__(self, base_path, force_reparse=False):
        self.base_path = base_path
        self.inner_path = "data/lisa/data/timit/raw/TIMIT"
        self.full_path = os.path.join(self.base_path, self.inner_path)
        self.force_reparse = force_reparse

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

        # First parse speakers and sentences, then build dictionary
        self._parse_spkr_info()
        self._parse_spkr_sent()
        self._parse_word_and_phon_ids()

    def _lex_word_and_phoneme_counts(self):
        # Parse lexicon words and phonemes
        self.lex_phon_count = Counter()
        self.lex_word_count = Counter()
        with open(self.dict_path, "r") as f:
            for line in f:
                if line[0] == ";":
                    continue
                word, phonemes = line.split("  ")
                phonemes = [x.strip().replace("/", "") for x in phonemes[1:-1].split()]
                self.lex_word_count += Counter([word])
                self.lex_phon_count += Counter(phonemes)

    def _trans_word_and_phoneme_counts(self):
        # There are phonemes that do not occur in the lexicon therefore
        # we must parse all of the sentences to build a dictionary
        self.trans_phon_count = Counter()
        self.trans_word_count = Counter()
        for spkr, data in self.spkr_sents.items():
            sents = [s_type + i for s_type, ids in data.items() for i in ids]
            for sent in sents:
                _, wrds, phons = self.get_sentence_data(spkr, sent)
                self.trans_word_count += Counter([wrd for _, _, wrd in wrds])
                self.trans_phon_count += Counter([phn for _, _, phn in phons])


    def _parse_word_and_phon_ids(self):
        if os.path.exists("cache/") and not self.force_reparse:
            with open("cache/all_word_counts.pickle", "rb") as f:
                self.all_word_count = pickle.load(f)
            with open("cache/all_phon_counts.pickle", "rb") as f:
                self.all_phon_count = pickle.load(f)
        else:
            self._lex_word_and_phoneme_counts()
            self._trans_word_and_phoneme_counts()
            self.all_phon_count = self.trans_phon_count + self.lex_phon_count
            self.all_word_count = self.trans_word_count + self.lex_word_count
            os.mkdir("cache/")
            with open("cache/all_word_counts.pickle", "wb") as f:
                pickle.dump(self.all_word_count, f)
            with open("cache/all_phon_counts.pickle", "wb") as f:
                pickle.dump(self.all_phon_count, f)
        self.word_to_id = {
            word: i for i, (word, _) in enumerate(self.all_word_count.most_common())
        }
        self.phon_to_id = {
            phon: i for i, (phon, _) in enumerate(self.all_phon_count.most_common())
        }
        self.pid_to_phon = {v, k for k, v in self.phon_to_id}
        self.wid_to_word = {v, k for k, v in self.word_to_id}

    def _parse_spkr_sent(self):
        self.spkr_sents = {}
        with open(self.spkr_sent_path, "r") as f:
            for line in f:
                if line[0] == ";":
                    continue
                line = [x.strip() for x in line.split()]
                self.spkr_sents[line[0]] = {
                    "SA" : line[1:3],
                    "SX" : line[3:8],
                    "SI" : line[8:11]
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
            "num_all_words": len(self.all_word_count.most_common()),
            "num_all_phons": len(self.all_phon_count.most_common())
        }

    def _wav(self, file_path):
        return wavfile.read(file_path)

    def _wrd(self, file_path):
        wrd_list = []
        with open(file_path, "r") as f:
            for line in f:
                s, e, wrd = line.split()
                s, e = int(s), int(e)
                wrd_list.append((s, e, wrd))
        return wrd_list

    def _phn(self, file_path):
        phn_list = []
        with open(file_path, "r") as f:
            for line in f:
                s, e, phn = line.split()
                s, e = int(s), int(e)
                phn_list.append((s, e, phn))
        return phn_list

    def get_sentence_data(self, speaker, sid):
        # Because the TIMIT dataset uses the NIST SPHERE header we
        # first convert it into a standard WAV if we have not already
        spkr = self.speakers[speaker]
        part = "TRAIN" if spkr["use"] == "TRN" else "TEST"
        gend = spkr["sex"]
        dr = "DR" + spkr["dr"]
        folder = os.path.join(self.full_path, part, dr, gend + speaker)
        phn_file = os.path.join(folder, sid + ".PHN")
        sph_wav_file = os.path.join(folder, sid + ".WAV")
        wav_file = os.path.join(folder, sid + ".REALWAV")
        if not os.path.exists(wav_file):
            sph = SPHFile(sph_wav_file)
            sph.write_wav(wav_file)
        wrd_file = os.path.join(folder, sid + ".WRD")
        return self._wav(wav_file), self._wrd(wrd_file), self._phn(phn_file)

if __name__ == "__main__":
    timit = TIMITDataset('./TIMIT')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(timit.stats())
    pp.pprint(timit.get_sentence_data("CJF0", "SA1"))
