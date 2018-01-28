from timit import TIMITDataset
from vrae import VRAE
from signal_utils import spectrogram
import numpy as np

if __name__ == "__main__":
    timit = TIMITDataset('./TIMIT')
    #timit.preprocess_spectrograms()
    batch_size = 10
    vrae = VRAE(
        input_size=512,
        batch_size=batch_size,
        latent_size=20
    )

    fft_size = 512
    step_size = 16
    thresh = 4

    batch = timit.batch_generator(batch_size)
    avg_loss = 0
    for i in range(100):
        batch_input = [example[0] for example in next(batch)]
        max_seq_length = max(x.shape[0] for x in batch_input)
        bi_arr = np.zeros((batch_size, max_seq_length, 512))
        for j, example in enumerate(batch_input):
            bi_arr[j, :example.shape[0], :example.shape[1]] = example
        loss, kl, rloss = vrae.train_batch(bi_arr)
        avg_loss = (avg_loss * i + loss) / (i + 1)
        print("Batch {}, loss: {}, kl:{}, rloss:{}, avg_loss:{}".format(i, loss, kl, rloss, avg_loss))
    vrae.save()
