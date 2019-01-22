# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess.py ]
#   Synopsis     [ preprocess text transcripts and audio speech of the LectureDSP dataset for the Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
import nltk
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from utils import data
from functools import partial
from pypinyin import Style, pinyin
from config import config, get_preprocess_args
from concurrent.futures import ProcessPoolExecutor


################
# PROCESS TEXT #
################
def process_text(mapper, input_path, output_path):
	input_file = []
	with open(input_path, 'r', encoding='utf-8') as r:
		lines = r.readlines()
		input_file = [ line.split() for line in lines ]

	with open(output_path, 'w', encoding='utf-8') as w:
		for line in input_file:
			w.write(line[0] + ' ')
			write = ''
			for word in line[1:]:
				for char in word.split(']['):
					try:	
						write += mapper[char.strip('[').strip(']')]
					except:
						write += word
				write += ' '
			w.write(write.strip() + '\n')


#################
# PROCESS AUDIO #
#################
"""
	Preprocesses the audio files from given input path into given output directories.
	Use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
	can omit it and just call _process_utterance on each input if you want.

	Args:
		transcript_path: The path to the transcript file
		input_dir: The directory where the audio is contained
		output_dir: The directory to write the output processed audio to
		visualization_dir: The directory to write the output plot to
		file_suffix: Filename extension of audio files
		start_from: The file index to start processing, if the preprocess progress is interupt, use this index to resume
		num_workers: Optional number of worker processes to parallelize across
		tqdm: You can optionally pass tqdm to get a nice progress bar
		vis_process: visualize the preprocess effect
		vis_origin: visualize the original waveform

	Returns:
		A list of tuples describing the training examples. This should be written to meta_text.txt by 'write_meta_data'
"""
def process_audio(input_dir, output_dir, visualization_dir, target_dBFS,
				  file_suffix='.wav', start_from=0, 
				  num_workers=1, tqdm=lambda x: x,
				  vis_process=False, vis_origin=False):

	if not os.path.isdir(input_dir):
		raise ValueError('Please make sure there are .wav files in the directory: ', input_dir)
	if os.path.isdir(output_dir) or os.path.isdir(visualization_dir):
		print('Output directories already exist, please remove these existing directories to proceed.')
		while True:
			proceed = str(input('Proceed? [y/n]: '))
			if proceed == 'y': break
			elif proceed == 'n': return
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if not os.path.isdir(visualization_dir):
		os.makedirs(visualization_dir)	

	wavs = sorted(glob.glob(os.path.join(input_dir, '*' + file_suffix)))
	
	if vis_origin:
		for wav in wavs:
			y, sr = librosa.load(wav)
			data.visualization(wav.split('/')[-1].split('.')[0], y, None, sr, output_dir, visualization_dir, vis_process=False)
	
	else:
		executor = ProcessPoolExecutor(max_workers=num_workers)
		futures = []

		for i, wav in enumerate(tqdm(wavs)):
			if i == 0: 
				y, sr = librosa.load(wav)
				print('Sample rate: ', sr)
			if i + 1 >= start_from:
				futures.append(executor.submit(partial(data.apply_audio_preprocess, wav, target_dBFS, file_suffix, output_dir, visualization_dir, vis_process)))

		for future in tqdm(futures): future.result()
		print('Progress: %i/%i: Complete!' % (len(wavs), len(wavs)))


##################
# PROCESS PINYIN #
##################
def process_pinyin(text_pinyin_path, text_input_raw_path, text_dir, all_text_output_path, text_input_file_list, join=False):
	
	if join:
		with open(os.path.join(text_dir, all_text_output_path), 'w', encoding='utf-8') as w:
			for input_path in text_input_file_list:
				input_path = os.path.join(text_dir, input_path)
				with open(input_path, 'r', encoding='utf-8') as r:
					lines = r.readlines()
					for line in lines: 
						w.write(line)

	def _ch2pinyin(txt_ch):
		ans = pinyin(txt_ch, style=Style.TONE2, errors=lambda x: x, strict=False)
		return [x[0] for x in ans if x[0] != 'EMPH_A']
	
	with open(text_pinyin_path, 'w') as w:
		with open(text_input_raw_path, 'r') as r:
			lines = r.readlines()
			for line in lines:
				tokens = line[:-1].split(' ')
				wid, txt_ch = tokens[0], ' '.join(_ch2pinyin(tokens[1:]))
				w.write(wid + '|' + txt_ch + '\n')


#############
# MAKE META #
#############
def make_meta(text_pinyin_path, input_wav_dir, meta_audio_dir, meta_text, num_workers, frame_shift_ms):
	os.makedirs(meta_audio_dir, exist_ok=True)
	metadata = data.build_from_path(text_pinyin_path, input_wav_dir, meta_audio_dir, num_workers, tqdm=tqdm)
	data.write_meta_data(metadata, meta_audio_dir, meta_text, frame_shift_ms)


####################
# DATASET ANALYSIS #
####################
def dataset_analysis(wav_dir, text_dir, text_input_file_list):
	nltk.download('wordnet')
	
	all_text = []
	all_audio = []
	for input_path in text_input_file_list:
		input_path = os.path.join(text_dir, input_path)
		with open(input_path, 'r', encoding='utf-8') as r:
			lines = r.readlines()
			for line in lines: 
				all_text.append(line.split()[1:])
				all_audio.append(line.split()[0])
	print('Training data count: ', len(all_text))
	
	line_switch_count = 0
	total_switch_count = 0
	
	for line in all_text:
		for text in line:
			if nltk.corpus.wordnet.synsets(text): total_switch_count += 1
		for text in line:
			if nltk.corpus.wordnet.synsets(text): 
				line_switch_count += 1
				break
	print('Total number of switches: ', total_switch_count)
	print('Total number of sentences containing a switch: ', line_switch_count)

	duration = 0.0
	max_d = 0
	min_d = 60
	for audio in tqdm(all_audio):
		y, sr = librosa.load(os.path.join(wav_dir, audio + '.wav'))
		d = librosa.get_duration(y=y, sr=sr)
		if d > max_d: max_d = d
		if d < min_d: min_d = d
		duration += d
	print('Switch frequency - total number of switch / hour: ', total_switch_count / (duration / 60**2))
	print('Speech total length (hr): ', duration / 60**2)
	print('Max duration (seconds): ', max_d)
	print('Min duration (seconds): ', min_d)
	print('Average duration (seconds): ', duration / len(all_audio))


########
# MAIN #
########
def main():

	args = get_preprocess_args()
	
	#---preprocess text---#
	if args.mode == 'all' or args.mode == 'text':
		try:
			mapper = data.get_mapper(os.path.join(args.text_dir, args.mapper_path))
			process_text(mapper, input_path=os.path.join(args.text_dir, args.text_input_train_path), output_path=os.path.join(args.text_dir, args.text_output_train_path))
			process_text(mapper, input_path=os.path.join(args.text_dir, args.text_input_dev_path), output_path=os.path.join(args.text_dir, args.text_output_dev_path))
			process_text(mapper, input_path=os.path.join(args.text_dir, args.text_input_test_path), output_path=os.path.join(args.text_dir, args.text_output_test_path))
		except: pass
		process_pinyin(args.text_pinyin_path, 
					   args.text_input_raw_path, 
					   args.text_dir, 
					   args.all_text_output_path, 
					   [args.text_output_train_path, 
					   args.text_output_dev_path, 
					   args.text_output_test_path], 
					   join=args.join)		

	#---preprocess audio---#
	elif args.mode == 'all' or args.mode == 'audio':
		process_audio(args.audio_input_dir, 
					  args.audio_output_dir, 
					  args.visualization_dir, 
					  target_dBFS=-10.0,
					  file_suffix='.wav', 
					  start_from=0,
					  num_workers=args.num_workers, 
					  tqdm=tqdm,
					  vis_process=True, 
					  vis_origin=False)
		data.check(args.audio_input_dir, args.audio_output_dir, file_suffix='*.wav')

	#---preprocess text and data to be model ready---#
	elif args.mode == 'all' or args.mode == 'meta':
		make_meta(args.text_pinyin_path, args.audio_output_dir, args.meta_audio_dir, args.meta_text, args.num_workers, config.frame_shift_ms)

	#---dataset analysis---#
	elif args.mode == 'all' or args.mode == 'analysis':
		dataset_analysis(args.audio_input_dir, args.text_dir, [args.text_output_train_path, args.text_output_dev_path, args.text_output_test_path])
	
	else:
		raise RuntimeError('Invalid mode!')



if __name__ == '__main__':
	main()