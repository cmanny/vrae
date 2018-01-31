from timit import TIMITDataset
from vrae import VRAE
from signal_utils import spectrogram
import numpy as np
import time

if __name__ == "__main__":
    timit = TIMITDataset('./TIMIT', fft_size=128, window_size=2048, thresh=3)
    #timit.preprocess_spectrograms()
    batch_size = 10
    vrae = VRAE(
        input_size=128,
        batch_size=batch_size,
        latent_size=64,
        learning_rate=0.001,
        save_path="ckpt/sa_batches1",
        num_layers=3,
        hidden_size=64
    )

    fft_size = 512
    step_size = 16
    thresh = 4

    batch = timit.batch_generator(batch_size)
    avg_loss = 0
    num_batches = 1 * 600*2 // batch_size
    for i in range(num_batches):
        start_time = time.time()
        batch_input = [example[0][:,:] for example in next(batch)]
        max_seq_length = max(x.shape[0] for x in batch_input)
        bi_arr = np.zeros((batch_size, max_seq_length, 128))
        for j, example in enumerate(batch_input):
            bi_arr[j, :example.shape[0], :example.shape[1]] = example
        loss, kl, rloss = vrae.train_batch(bi_arr, [max_seq_length]*batch_size, i)
        avg_loss = (avg_loss * i + loss) / (i + 1)
        diff = time.time() - start_time
        est_time_remaining = diff * (num_batches - (i + 1)) / 60
        print("Estimated time remaining: {}m".format(est_time_remaining))
        print(
            "Batch {}/{}, loss: {}, kl:{}, rloss:{}, avg_loss:{}".format(
                i, num_batches, loss, kl, rloss, avg_loss
            )
        )
