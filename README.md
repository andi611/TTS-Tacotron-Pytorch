# Tacotron
A Pytorch implementation of [Google's Tacotron](https://arxiv.org/pdf/1703.10135.pdf) speech synthesis network.

This implementation includes pre-trained model and uses the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).

<img src="https://i.imgur.com/zHp0fNS.png" width="1044" height="537">

## Introduction
This work is based on [r9y9/tacotron_pytorch](https://github.com/r9y9/tacotron_pytorch), the main differences are:
* Adds **location-sensitive attention** and the **stop token** from the [Tacotron 2](https://arxiv.org/pdf/1712.05884.pdf) paper.
  This can greatly reduce the amount of time and data required to train a model.
* Remove all TensorFlow dependencies that [r9y9](https://github.com/r9y9/tacotron_pytorch) uses, now it runs on PyTorch and PyTorch only.
* Adds a [loss](model/loss.py) module, and use MSE loss instead of L1 loss.
* Adds a [data loader](dataloader.py) module.
* Incorporate the LJ Speech data preprocessing script from [keithito](https://github.com/keithito/tacotron).
* Code factoring and optimization for easier debug and extend in the furture.

The main differences from the original [Tacotron]((https://arxiv.org/pdf/1703.10135.pdf)) paper:
* TODO

Audio quality isn't as good as Google's demo yet, but hopefully it will improve eventually. Pull requests are welcome!


## Quick Start

### Setup
* Clone this repo: `git clone git@github.com:andi611/Tacotron-Pytorch.git`
* CD into this repo: `cd Tacotron-Pytorch`

### Installing dependencies

1. Install Python 3.

2. Install the latest version of **[Pytorch](https://pytorch.org/get-started/locally/)** according to your platform. For better
	performance, install with GPU support (CUDA) if viable. This code works with Pytorch 0.4 and later.

3. Install [requirements](requirements.txt):
	```
	pip3 install -r requirements.txt
	```
	*Warning: you need to install torch depending on your platform. Here list the Pytorch version used when built this project was built.*


### Training

1. **Download the LJ Speech dataset.**
	* [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
	
	You can use other datasets if you convert them to the right format. See [TRAINING_DATA.md](https://github.com/keithito/tacotron/blob/master/TRAINING_DATA.md) for more info.

2. **Unpack the dataset into `~/Tacotron-Pytorch/data`**

	After unpacking, your tree should look like this for LJ Speech:
	```
 |- Tacotron-Pytorch
	 |- data
		 |- LJSpeech-1.1
			 |- metadata.csv
			 |- wavs
	```

3. **Preprocess the LJ Speech dataset and make model-ready meta files using [preprocess.py](preprocess.py):**
	```
	python3 preprocess.py --mode make
	```

	After preprocessing, your tree will look like this:
	```
 |- Tacotron-Pytorch
	 |- data
		 |- LJSpeech-1.1 (The downloaded dataset)
			 |- metadata.csv
			 |- wavs
		 |- meta (generate by preprocessing)
			 |- meta_text.txt 
			 |- meta_mel_xxxxx.npy ...
			 |- meta_spec_xxxxx.npy ...
		 |- test_transcripts.txt (provided)
	```

4. **Train a model using [train.py](train.py)**
	```
	python3 train.py --ckpt_dir ckpt/ --log_dir log/
	```

	Restore training from a previous checkpoint:
	```
	python3 train.py --ckpt_dir ckpt/ --log_dir log/ --model_name 480000
	```

	Tunable hyperparameters are found in [config.py](config.py). 
	
	You can adjust these parameters and setting by editing the file, the default hyperparameters are recommended for LJ Speech.

5. **Monitor with Tensorboard** (OPTIONAL)
	```
	tensorboard --logdir 'path to log_dir'
	```

	The trainer dumps audio and alignments every 2000 steps by default. You can find these in `tacotron/ckpt/`.


### Testing: Using a pre-trained model and [test.py](test.py)
* **Run the testing environment with interactive mode**:
	```
	python3 test.py --interactive --plot --model_name 470000
	```
* **Run the testing algorithm on a set of transcripts** (Results can be found in the [result/480000](result/480000) directory) :
	```
	python3 test.py --plot --model_name 480000 --test_file_path ../data/text/test_sample.txt
	```


## Acknowledgement
Credits to Ryuichi Yamamoto for a wonderful Pytorch implementation of [Tacotron](https://github.com/r9y9/tacotron_pytorch), which this work is  mainly based on.

## TODO
* Add more configurable hparams
* provide trained models and audio samples

