# APCodec: A Neural Audio Codec with Parallel Amplitude and Phase Spectrum Encoding and Decoding
### Yang Ai, Xiao-Hang Jiang, Ye-Xin Lu, Hui-Peng Du, Zhen-Hua Ling

In our [paper](https://arxiv.org/abs/2402.10533), 
we proposed APCodec: A novel neural audio codec which regards audio amplitude and phase spectra as coding objects.<br/>
We provide our implementation as open source in this repository.

**Abstract :**
This paper introduces a novel neural audio codec targeting high waveform sampling rates and low bitrates named APCodec, which seamlessly integrates the strengths of parametric codecs and waveform codecs. The APCodec revolutionizes the process of audio encoding and decoding by concurrently handling the amplitude and phase spectra as audio parametric characteristics like parametric codecs. It is composed of an encoder and a decoder with the modified ConvNeXt v2 network as the backbone, connected by a quantizer based on the residual vector quantization (RVQ) mechanism. The encoder compresses the audio amplitude and phase spectra in parallel, amalgamating them into a continuous latent code at a reduced temporal resolution. This code is subsequently quantized by the quantizer. Ultimately, the decoder reconstructs the audio amplitude and phase spectra in parallel, and the decoded waveform is obtained by inverse short-time Fourier transform. To ensure the fidelity of decoded audio like waveform codecs, spectral-level loss, quantization loss, and generative adversarial network (GAN) based loss are collectively employed for training the APCodec. To support low-latency streamable inference, we employ feed-forward layers and causal deconvolutional layers in APCodec, incorporating a knowledge distillation training strategy to enhance the quality of decoded audio. Experimental results confirm that our proposed APCodec can encode 48 kHz audio at bitrate of just 6 kbps, with no significant degradation in the quality of the decoded audio. At the same bitrate, our proposed APCodec also demonstrates superior decoded audio quality and faster generation speed compared to well-known codecs, such as Encodec, AudioDec and DAC.

Visit our [demo website](https://yangai520.github.io/APCodec/) for audio samples.

## Requirements
```
torch==1.8.1+cu111
numpy==1.21.6
librosa==0.9.1
tensorboard==2.8.0
soundfile==0.10.3
matplotlib==3.1.3
```

## Data Preparation
For training, write the list paths of training set and validation set to `input_training_wav_list` and `input_validation_wav_list` in `config.json`, respectively.

For inference, write the test set waveform path to `test_input_wavs_dir` in `config.json`.

**Note :** The recommended sampling rate for this code is 48kHz.

## Training
Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Inference:
Write the checkpoint path to `checkpoint_file_load_Encoder` and `checkpoint_file_load_Decoder` in `config.json`.

Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python inference.py
```
Run using CPU:
```
CUDA_VISIBLE_DEVICES=CPU python inference.py
```

## Citation
```
@article{ai2024apcodec,
  title={A{PC}odec: A Neural Audio Codec with Parallel Encoding and Decoding for Amplitude and Phase Spectra},
  author={Ai, Yang and Jiang, Xiao-Hang and Lu, Ye-Xin and Du, Hui-Peng and Ling, Zhen-Hua},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={32},
  pages={3256--3269},
  year={2024}
}
```
