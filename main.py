from timit import TIMITDataset
from vrae import VRAE
from signal_utils import spectrogram
import numpy as np
import time

if __name__ == "__main__":
    timit = TIMITDataset('./TIMIT', fft_size=512, window_size=512, thresh=3)
    #timit.preprocess_spectrograms()
    batch_size = 32
    vrae = VRAE(
        input_size=512,
        batch_size=batch_size,
        latent_size=10,
        learning_rate=0.001,
        save_path="ckpt/sa_32_4lay_mul_states_512_lr001_lat10_full_kl",
        num_layers=4,
        hidden_size=128
    )

    fft_size = 512
    step_size = 16
    thresh = 4

    batch = timit.batch_generator(batch_size)
    avg_loss = 0
    num_batches = 100 * 600*2 // batch_size
    for i in range(num_batches):
        start_time = time.time()
        current_batch = next(batch)
        batch_input = []
        for x in current_batch:
            phns = x[2]
            start = phns[1][0] // 512
            end = phns[-2][1] // 512
            batch_input.append(x[0][start:end,:])
        lengths = [x.shape[0] for x in batch_input]
        max_seq_length = max(lengths)
        bi_arr = np.zeros((batch_size, max_seq_length, 512))
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
