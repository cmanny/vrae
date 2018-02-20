from sklearn.decomposition import TruncatedSVD
import signal_utils
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row
import numpy as np

from scipy.io import wavfile

def mmse_spectrograms():
    out_list = []
    l = [
        "mmse/callum_sentence.wav",
        "mmse/raul_sentence.wav",
        "mmse/niall_sentence.wav",
        "mmse/callum_no_ifs_ands_buts.wav",
        "mmse/raul_no_ifs_ands_buts.wav",
        "mmse/niall_no_ifs_ands_buts.wav",
    ]
    for x in l:
        wav = wavfile.read(x)[1]
        wav_spectrogram = signal_utils.spectrogram(
            wav.astype('float64'),
            fft_size=1024,
            step_size=512,
            log=True,
            thresh=3
        )
        out_list.append(wav_spectrogram)
    return out_list

def pca_mmse():
    def mscatter(p, x, y, color=None):
        p.scatter(x, y, size=1, line_color=color, fill_color=color)
    p = figure(title="mmse", plot_width=600, plot_height=600)
    p.grid.grid_line_color = None
    p.background_fill_color = "#eeeeee"

    specs = mmse_spectrograms()
    print(specs[0])
    colors = ["red", "green", "blue", "orange", "yellow", "purple"]
    all_vecs = np.concatenate(specs, axis=0)
    svd = TruncatedSVD(n_components=2).fit(all_vecs)
    for c, spec in zip(colors, specs):
        pcas = svd.transform(spec)
        mscatter(p, pcas[:, 0], pcas[:, 1], color=c)

    output_file("mmse_pcas.html", title="zs test")

    show(row(p))  # open a browser

if __name__ == "__main__":
    pca_mmse()
