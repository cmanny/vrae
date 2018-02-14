from timit import TIMITDataset
from vrae import VRAE
from signal_utils import spectrogram
import numpy as np
import time
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import row

from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from scipy.io import wavfile

def pca_z(z_run):
    svd = TruncatedSVD(n_components=2).fit(z_run)
    pca_z = svd.transform(z_run)
    return pca_z

def tsne_z(z_run):
    tSNE_model = TSNE(verbose=2, perplexity=80, min_grad_norm=1E-12, n_iter=3000)
    z_run_tsne = tSNE_model.fit_transform(z_run)
    ax1[1].scatter(z_run_tsne[:, 0], z_run_tsne[:, 1], c=label, marker='*', linewidths=0)
    ax1[1].set_title('tSNE on z_run')

def latent_spaces(zs):
    pca_zs = [pca_z_run(z) for z in zs]


def train():
    window_size = 512
    fft_size = 512
    timit = TIMITDataset('./TIMIT',
        fft_size=fft_size,
        window_size=window_size,
        thresh=3
    )
    batch_size = 32
    vrae = VRAE(
        input_size=fft_size,
        batch_size=batch_size,
        latent_size=8,
        learning_rate=0.001,
        save_path="ckpt/several_latent_spaces",
        num_layers=4,
        hidden_size=128,
        keep_prob=0.9
    )

    thresh = 4

    batch = timit.batch_generator(batch_size)
    avg_loss = 0
    num_batches = 10 * 6000*2 // batch_size
    for i in range(num_batches):
        start_time = time.time()
        current_batch = next(batch)
        batch_input = []
        for x in current_batch:
            phns = x[2]
            start = phns[1][0] // window_size
            end = phns[-2][1] // window_size
            batch_input.append(x[0][start:end,:])
        lengths = [x.shape[0] for x in batch_input]
        max_seq_length = max(lengths)
        bi_arr = np.zeros((batch_size, max_seq_length, fft_size))
        for j, example in enumerate(batch_input):
            normed_ex = (example - np.min(example)) / (np.max(example) - np.min(example))
            bi_arr[j, :example.shape[0], :example.shape[1]] = normed_ex
        loss, kl, rloss = vrae.train_batch(bi_arr, lengths, i)
        avg_loss = (avg_loss * i + loss) / (i + 1)
        diff = time.time() - start_time
        est_time_remaining = diff * (num_batches - (i + 1)) / 60
        print("Estimated time remaining: {}m".format(est_time_remaining))
        print(
            "Batch {}/{}, loss: {}, kl:{}, rloss:{}, avg_loss:{}".format(
                i, num_batches, loss, kl, rloss, avg_loss
            )
        )
        if i % 500 == 0:
            vrae.save(i)
    vrae.save(i)

def mmse_spectrograms():
    out_list = []
    l = [
        "mmse/callum_sentence.wav",
        "mmse/raul_sentence.wav",
        "mmse/niall_sentence.wav"
    ]
    for x in l:
        wav = wavfile.read(x)[1]
        wav_spectrogram = spectrogram(
            wav.astype('float64'),
            fft_size=1024,
            step_size=512,
            log=True,
            thresh=3
        )
        out_list.append(wav_spectrogram)
    return out_list

def explore_latent_space():
    window_size = 512
    fft_size = 512
    timit = TIMITDataset('./TIMIT',
        fft_size=fft_size,
        window_size=window_size,
        thresh=3
    )
    batch_size = 32
    vrae = VRAE(
        input_size=fft_size,
        batch_size=batch_size,
        latent_size=8,
        learning_rate=0.001,
        save_path="ckpt/several_latent_spaces",
        num_layers=4,
        hidden_size=128,
        keep_prob=0.9
    )

    thresh = 4

    batch = timit.batch_generator(batch_size)
    num_batches = 20
    vrae.load("ckpt/several_latent_spaces-3749")
    all_zs = [np.empty((0, 8))]*5

    def mscatter(p, x, y, color=None):
        p.scatter(x, y, size=1, line_color=color, fill_color=color)

    ps = [0 for _ in range(5)]
    for i in range(5):
        p = figure(title="z_{}".format(i), plot_width=400, plot_height=400)
        p.grid.grid_line_color = None
        p.background_fill_color = "#eeeeee"
        ps[i] = p

    spkr_info_list = []

    for i in range(num_batches):
        current_batch = next(batch)
        batch_input = []
        for xb in current_batch:
            x = xb[0]
            spkr_info_list.append(xb[1])
            phns = x[2]
            start = phns[1][0] // window_size
            end = phns[-2][1] // window_size
            batch_input.append(x[0][start:end,:])
        lengths = [x.shape[0] for x in batch_input]
        max_seq_length = max(lengths)
        bi_arr = np.zeros((batch_size, max_seq_length, fft_size))
        for j, example in enumerate(batch_input):
            normed_ex = (example - np.min(example)) / (np.max(example) - np.min(example))
            bi_arr[j, :example.shape[0], :example.shape[1]] = normed_ex
        zs = vrae.recognize(bi_arr, lengths)
        for i, z in enumerate(zs):
            all_zs[i] = np.append(all_zs[i], z, axis=0)
    pca_runs = []
    for zs in all_zs:
        pca_runs.append(pca_z(zs))
    # for p, pcazs in zip(ps, pca_runs):
    #     male = np.empty((0, 2))
    #     female = np.empty((0, 2))
    #     for i, pcaz in enumerate(pcazs):
    #         if spkr_info_list[i]["sex"] == "M":
    #             male = np.append(male, [pcaz], axis=0)
    #         else:
    #             female = np.append(female, [pcaz], axis=0)
    #     mscatter(p, male[:, 0], male[:, 1], color="blue")
    #     mscatter(p, female[:, 0], female[:, 1], color="red")
    for p, pcazs in zip(ps, pca_runs):
        mscatter(p, pcazs[:, 0], pcazs[:, 1], color="yellow")

    specs = mmse_spectrograms()
    colors = ["red", "green", "blue"]
    for i in range(4):
        for c, spec in zip(colors, specs):
            normed_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
            batch = np.repeat([normed_spec], 32, axis=0)
            lengths = [x.shape[0] for x in batch]
            all_zs = vrae.recognize(bi_arr, lengths)
            for p, zs in zip(ps, all_zs):
                pcazs = pca_z(zs)
                mscatter(p, pcazs[:, 0], pcazs[:, 1], color=c)

    output_file("mmse_sentence.html", title="zs test")

    show(row(*ps))  # open a browser


if __name__ == "__main__":
    explore_latent_space()
