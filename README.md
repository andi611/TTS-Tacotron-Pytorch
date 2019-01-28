# Tacotron
A Pytorch implementation of [Google's Tacotron](https://arxiv.org/pdf/1703.10135.pdf) speech synthesis network, with pre-trained model.

<img src="https://i.imgur.com/zHp0fNS.png" width="1044" height="537">

## Introduction
This work is based on [r9y9/tacotron_pytorch](https://github.com/r9y9/tacotron_pytorch), the main differences are:
* Adds the **stop token** from the [Tacotron 2](https://arxiv.org/pdf/1712.05884.pdf) paper.
  This can greatly reduce the amount of data required to train a model, and is refer to as `gate` in this implementation.
* Remove all TensorFlow dependencies that [r9y9](https://github.com/r9y9/) uses, now it runs on PyTorch and PyTorch only.
* Adds a [loss](model/loss.py) module, and use MSE loss instead of L1 loss.
* Adds a [data loader](dataloader.py) module.
* Incorporate the LJ Speech data preprocessing script from [keithito](https://github.com/keithito/tacotron).
* Code factoring and optimization for easier debug and extend in the furture.

The main differences from the original [Tacotron]((https://arxiv.org/pdf/1703.10135.pdf)) paper:
* TODO

Audio quality isn't as good as Google's demo yet, but hopefully it will improve eventually. Pull requests are welcome!


## Quick Start

### Installing dependencies

1. Install Python 3.

2. Install the latest version of **[Pytorch](https://pytorch.org/get-started/locally/)** according to your platform. For better
	performance, install with GPU support (CUDA) if viable. This code works with Pytorch 0.4 and later.

3. Install [requirements](requirements.txt):
	```
	pip3 install -r requirements.txt
	```
	*Warning: you need to install torch depending on your platform. Here list the Pytorch version used when built this project was built.*


### Using a pre-trained model
* **Run the testing environment with interactive mode**:
	```
	python3 test.py --interactive --plot --model 470000
	```
* **Run the testing algorithm on a set of transcripts** (Results can be found in the [result/480000](result/480000) directory) :
	```
	python3 test.py --plot --model 480000 --test_file_path ../data/text/test_sample.txt
	```


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
	python3 train.py
	```

	Tunable hyperparameters are found in [config.py](config.py). 
	
	You can adjust these parameters and setting by editing the file, the default hyperparameters are recommended for LJ Speech.

6. **Monitor with TensorboardX** (optional)
	```
	tensorboard --logdir 'path to log dir'
	```

	The trainer dumps audio and alignments every 2000 steps by default. You can find these in `tacotron/ckpt`.


## Acknowledgement
Credits to Ryuichi Yamamoto for a wonderful Pytorch implementation of [Tacotron](https://github.com/r9y9/tacotron_pytorch), which this work is  mainly based on.

