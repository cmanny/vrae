from timit import TIMITDataset
from vrae import VRAE
from signal_utils import spectrogram
import numpy as np

if __name__ == "__main__":
    timit = TIMITDataset('./TIMIT')
    batch_size = 2
    vrae = VRAE(
        input_size=512,
        batch_size=batch_size,
        latent_size=20
    )

    fft_size = 512
    step_size = 16
    thresh = 4

    batch = timit.batch_generator(batch_size)
    for i in range(1):
        next_batch = next(batch)
        batch_input = []
        for example in next_batch:
            wav = example[0][1]
            wav_spectrogram = spectrogram(
                wav.astype('float64'),
                fft_size=fft_size*2,
                step_size=step_size,
                log=True,
                thresh=thresh
            )
            batch_input.append(wav_spectrogram)
        max_seq_length = max(x.shape[0] for x in batch_input)
        bi_arr = np.zeros((batch_size, max_seq_length, 512))
        for i, example in enumerate(batch_input):
            bi_arr[i, :example.shape[0], :example.shape[1]] = example
        print("Batch {}, loss: {}, kl:{}, rloss:{}".format(i, *vrae.train_batch(bi_arr)))
    vrae.save()
