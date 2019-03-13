# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ preprocess.py ]
#   Synopsis     [ preprocess text transcripts and audio speech for the Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import glob
import librosa
import argparse
from utils import data
from tqdm import tqdm
from config import config, get_preprocess_args


#############
# MAKE META #
#############
def make_meta(text_input_path, audio_input_dir, meta_dir, meta_text, file_suffix, num_workers, frame_shift_ms):
	os.makedirs(meta_dir, exist_ok=True)
	metadata = data.build_from_path(text_input_path, audio_input_dir, meta_dir, file_suffix, num_workers, tqdm=tqdm)
	data.write_meta_data(metadata, meta_dir, meta_text, frame_shift_ms)


####################
# DATASET ANALYSIS #
####################
def dataset_analysis(wav_dir, file_suffix):

	audios = sorted(glob.glob(os.path.join(wav_dir, '*.' + file_suffix)))
	print('Training data count: ', len(audios))

	duration = 0.0
	max_d = 0
	min_d = 60
	for audio in tqdm(audios):
		y, sr = librosa.load(audio)
		d = librosa.get_duration(y=y, sr=sr)
		if d > max_d: max_d = d
		if d < min_d: min_d = d
		duration += d

	print('Sample rate: ', sr)
	print('Speech total length (hr): ', duration / 60**2)
	print('Max duration (seconds): ', max_d)
	print('Min duration (seconds): ', min_d)
	print('Average duration (seconds): ', duration / len(audios))


########
# MAIN #
########
def main():

	args = get_preprocess_args()

	if args.mode == 'all' or args.mode == 'make' or args.mode == 'analyze':
		
		#---preprocess text and data to be model ready---#
		if args.mode == 'all' or args.mode == 'make':
			make_meta(args.text_input_path, args.audio_input_dir, args.meta_dir, args.meta_text, args.file_suffix, args.num_workers, config.frame_shift_ms)

		#---dataset analyze---#
		if args.mode == 'all' or args.mode == 'analyze':
			dataset_analysis(args.audio_input_dir, args.file_suffix)
	
	else:
		raise RuntimeError('Invalid mode!')



if __name__ == '__main__':
	main()