# CS-Tacotron
A Pytorch implementation of CS-Tacotron, a code-switching speech synthesis end-to-end generative TTS model based on Tacotron.  

## Introduction
With the wide success of recent machine learning Text-to-speech (TTS) models, promising results on synthesizing realistic speech have proven machine’s capability of synthesizing human-like voices. However, little progress has been made in the domain of Chinese-English code-switching text-to-speech synthesis, where machine has to learn to handle both input and output in a multilingual fashion. Code-switching occurs when a speaker alternates between two or more languages, nowadays people communicates in code-switching languages in everyday life, hence spoken language technologies such as TTS must be developed to handle multilingual input and output.

In this work, we present Code-Switching Tacotron, which is built based on the state-of-the-art end-to-end text-to-speech generative model Tacotron (Wang et al., 2017). CS-Tacotron is capable of synthesizing code-switching speech conditioned on raw CS text. Given CS text and audio pairs, our model can be trained end-to-end with proper data pre-processing. Furthurmore, we train our model on the LectureDSP dataset, a Chinese-English code-switching lecture-based dataset, which originates from the course Digital Signal Processing (DSP) offered in National Taiwan University (NTU). We present several key implementation techniques to make the Tacotron model perform well on this challenging multilingual speech generation task. CS-Tacotron possess the capability of generating CS speech from CS text, and speaks vividly with the style of LectureDSP’s speaker.

See **[report.pdf](report.pdf)** for more detail of this work.

Pull requests are welcome!

## Demo
Audio samples of CS-Tacotron. All of the below phrases are unseen during training. 

* If you are reading this on Github, **please visit our [Github page](https://andi611.github.io/CS-Tacotron/) for the audio bars to display properly.**
* Audio files and their corresponding < spectrogram / alignment plots > can also be found in [result/](result/).

### CS-Tacotron works well on monolingual Chinese inputs. 
* <audio src="result/ch1.wav" controls preload></audio> - "這是數位語音處理"
* <audio src="result/ch2.wav" controls preload></audio> - "今天天氣很好"
* <audio src="result/ch3.wav" controls preload></audio> - "歡迎來到台灣大學"
* <audio src="result/ch4.wav" controls preload></audio> - "歡迎來到語音處理實驗室"
* <audio src="result/ch5.wav" controls preload></audio> - "吃什麼好呢"

### CS-Tacotron works well on **out-of-domain mixlingual Chinese-English** inputs. 
* <audio src="result/mix1.wav" controls preload></audio> - "每天都要 HAPPY"
* <audio src="result/mix2.wav" controls preload></audio> - "這是語音處理 PROCESSING"
* <audio src="result/mix3.wav" controls preload></audio> - "你可以多使用 GOOGLE"
* <audio src="result/mix4.wav" controls preload></audio> - "NEW YEAR 新氣象"
* <audio src="result/mix5.wav" controls preload></audio> - "這是個好 PROBLEM"

### CS-Tacotron can also adpat to some **out-of-domain monolingual English** inputs,
* despite the fact that none of the training data contains full English sentence. 
* <audio src="result/en1.wav" controls preload></audio> - "TAIWAN NUMBER ONE"
* <audio src="result/en2.wav" controls preload></audio> - "YOU HAVE SOME PROBLEM"

## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of **[Pytorch](https://pytorch.org/get-started/locally/)** according to your platform. For better
	performance, install with GPU support (CUDA) if viable. This code works with Pytorch 1.0 and later.

3. (Optional) Install the latest version of [Tensorflow](https://www.tensorflow.org/) according to your platform. This can be optional, but for now required for speech processing.

4. Install [requirements](requirements.txt):
	```
	pip3 install -r requirements.txt
	```
	*Warning: you need to install torch and tensorflow / tensorflow-gpu depending on your platform. Here we list the Pytorch and tensorflow version that we use when we built this project.*


### Using a pre-trained model
* **Run the testing environment with interactive mode**:
	```
	python3 test.py --interactive --plot --long_input --model 470000
	```
* **Run the testing algorithm on a set of transcripts** (Results can be found in the [result/480000](result/480000) directory) :
	```
	python3 test.py --plot --model 480000 --test_file_path ../data/text/test_sample.txt
	* '--long_input' is optional to add
	```


### Training

*Note: We trained our model on our own dataset: LectureDSP. Currently this dataset is not available for public release and remains a private collection in the lab. See 'report.pdf' for more information about this dataset.*

1. **Download a code-switch dataset of your choice.**

2. **Unpack the dataset into `~/data/text` and `~/data/audio`.**

	After unpacking, your data tree should look like this for the default paths to work:
	```
	./CS-Tacotron
	 |- data
		 |- text
		 	|- train_sample.txt
		 	|- test_sample.txt
		 |- audio
		 	|- sample 
		 		|- audio_sample_*.wav
		 		|- ...
	```

*Note: For the following section, set the paths according to the file names of your dataset, this is just a demonstration of some sample data. The format of your dataset should match the provided sample data for this code to work.*

3. **Preprocess the text data using [src/preprocess.py](src/preprocess.py):**
	```
	python3 preprocess.py --mode text --text_input_raw_path ../data/text/train_sample.txt --text_pinyin_path '../data/text/train_sample_pinyin.txt'
	```


4. **Preprocess the audio data using [src/preprocess.py](src/preprocess.py):**
	```
	python3 preprocess.py --mode audio --audio_input_dir ../data/audio/sample/ --audio_output_dir ../data/audio/sample_processed/ --visualization_dir ../data/audio/sample_visualization/
	```
	[Visualization](https://i.imgur.com/aqqJwtB.jpg) of the audio preprocess differences:
	![](https://i.imgur.com/aqqJwtB.jpg)

5. **Make model-ready meta files from text and audio using [src/preprocess.py](src/preprocess.py):**
	```
	python3 preprocess.py --mode meta --text_pinyin_path ../data/text/train_sample_pinyin.txt --audio_output_dir ../data/audio/sample_processed/
	```

5. **Train a model using [src/train.py](src/train.py)**
	```
	python3 train.py
	```

	Tunable hyperparameters are found in [src/config.py](src/config.py). You can adjust these parameters and setting by editing the file.
	The default hyperparameters are recommended for LectureDSP and other Chinese-English code switching data.


6. **Monitor with TensorboardX** (optional)
	```
	tensorboard --logdir 'path to log dir'
	```

	The trainer dumps audio and alignments every 2000 steps by default. You can find these in `CS-tacotron/ckpt`.


## Acknowledgement
We would like to give credit to the [work](https://github.com/r9y9/tacotron_pytorch) of Ryuichi Yamamoto, a wonderful Pytorch implementation of Tacotron, which we mainly based our work on.

## Alignment
We show the alignment plot of our model’s testing phase, where the first shows the alignment of monolingual Chinese input, the second is Chinese-English code-switching input, and the third is monolingual English input, respectively.
<img src="https://i.imgur.com/OSgJvvf.png" width="645" height="775">
