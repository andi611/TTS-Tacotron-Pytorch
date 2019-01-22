# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ config.py ]
#   Synopsis     [ configurations ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import argparse
from multiprocessing import cpu_count


########################
# MODEL CONFIGURATIONS #
########################
class configurations(object):
	
	def __init__(self):
		self.get_audio_config()
		self.get_model_config()
		self.get_dataloader_config()
		self.get_training_config()
		self.get_testing_config()

	def get_audio_config(self):
		self.num_mels = 80
		self.num_freq = 1025
		self.sample_rate = 22050
		self.frame_length_ms = 50
		self.frame_shift_ms = 12.5
		self.preemphasis = 0.97
		self.min_level_db = -100
		self.ref_level_db = 20
		self.hop_length = 250

	def get_model_config(self):
		self.embedding_dim = 256
		self.outputs_per_step = 5
		self.padding_idx = None
		self.use_memory_mask = False

	def get_dataloader_config(self):
		self.pin_memory = True
		self.num_workers = 2

	def get_training_config(self):
		self.batch_size = 16
		self.adam_beta1 = 0.9
		self.adam_beta2 = 0.999
		self.initial_learning_rate = 0.002
		self.decay_learning_rate = True
		self.nepochs = 1000
		self.weight_decay = 0.0
		self.clip_thresh = 1.0
		self.checkpoint_interval = 2000

	def get_testing_config(self):
		self.max_iters = 200
		self.griffin_lim_iters = 60
		self.power = 1.5 # Power to raise magnitudes to prior to Griffin-Lim

config = configurations()


###########################
# TRAINING CONFIGURATIONS #
###########################
def get_training_args():
	parser = argparse.ArgumentParser(description='training arguments')

	parser.add_argument('--checkpoint_dir', type=str, default='../ckpt_train', help='Directory where to save model checkpoints')
	parser.add_argument('--checkpoint_path', type=str, default=None, help='Restore model from checkpoint path if given')
	parser.add_argument('--data_root', type=str, default='../data/meta', help='Directory contains preprocessed features')
	parser.add_argument('--meta_text', type=str, default='meta_text.txt', help='Model-ready training transcripts')
	parser.add_argument('--summary_comment', type=str, default=None, help='Comment for log summary writer')

	args = parser.parse_args()
	return args


#############################
# PREPROCESS CONFIGURATIONS #
#############################
def get_preprocess_args():
	parser = argparse.ArgumentParser(description='preprocess arguments')

	parser.add_argument('--mode', choices=['text', 'audio', 'meta', 'analysis', 'all'], default='all', help='what to preprocess')
	parser.add_argument('--num_workers', type=int, default=cpu_count(), help='multi-thread processing')
	parser.add_argument('--join', type=bool, default=False, help='whether to join the [train, dev, test] transcripts into one joint transcript')

	meta_path = parser.add_argument_group('meta_path')
	meta_path.add_argument('--meta_audio_dir', type=str, default='../data/meta/', help='path to the model-ready training acoustic features')
	meta_path.add_argument('--meta_text', type=str, default='meta_text.txt', help='name of the model-ready training transcripts')
	
	audio_path = parser.add_argument_group('audio_path')
	audio_path.add_argument('--audio_input_dir', type=str, default='../data/audio/original/', help='directory path to the original audio data')
	audio_path.add_argument('--audio_output_dir', type=str, default='../data/audio/processed/', help='directory path to output the processed audio data')
	audio_path.add_argument('--visualization_dir', type=str, default='../data/audio/visualization/', help='directory path to output the audio visualization images')
	
	text_path = parser.add_argument_group('text_path')
	text_path.add_argument('--text_dir', type=str, default='../data/text/', help='directory to the text transcripts')
	text_path.add_argument('--mapper_path', type=str, default='mapper.txt', help='path to the encoding mapper')
	text_path.add_argument('--text_pinyin_path', type=str, default='../data/text/train_all_pinyin.txt', help='path to the transformed training text transcripts')

	input_path = parser.add_argument_group('text_input_path')
	input_path.add_argument('--text_input_train_path', type=str, default='train_ori.txt', help='path to the original training text data')
	input_path.add_argument('--text_input_dev_path', type=str, default='dev_ori.txt', help='path to the original development text data')
	input_path.add_argument('--text_input_test_path', type=str, default='test_ori.txt', help='path to the original testing text data')
	input_path.add_argument('--text_input_raw_path', type=str, default='../data/text/train_all.txt', help='path to the raw text transcripts')
	
	output_path = parser.add_argument_group('text_output_path')
	output_path.add_argument('--text_output_train_path', type=str, default='train.txt', help='path to the processed training text data')
	output_path.add_argument('--text_output_dev_path', type=str, default='dev.txt', help='path to the processed development text data')
	output_path.add_argument('--text_output_test_path', type=str, default='test.txt', help='path to the processed testing text data')
	output_path.add_argument('--all_text_output_path', type=str, default='train_all.txt', help='path to the joint processed text data')

	args = parser.parse_args()
	return args


#######################
# TEST CONFIGURATIONS #
#######################
def get_test_args():
	parser = argparse.ArgumentParser(description='testing arguments')

	parser.add_argument('--plot', action='store_true', help='whether to plot')
	parser.add_argument('--long_input', action='store_true', help='whether to set the model for long input')
	parser.add_argument('--interactive', action='store_true', help='whether to test in an interactive mode')

	path_parser = parser.add_argument_group('path')
	path_parser.add_argument('--result_dir', type=str, default='../result/', help='path to output test results')
	path_parser.add_argument('--ckpt_dir', type=str, default='../ckpt/', help='path to the directory where model checkpoints are saved')
	path_parser.add_argument('--checkpoint_name', type=str, default='checkpoint_step', help='model name prefix for checkpoint files')
	path_parser.add_argument('--model', type=str, default='480000', help='model step name for checkpoint files')
	path_parser.add_argument('--test_file_path', type=str, default='../data/text/test_sample.txt', help='path to the input test transcripts')
	
	args = parser.parse_args()
	return args

