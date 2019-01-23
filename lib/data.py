# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ data.py ]
#   Synopsis     [ utility functions for preprocess.py ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import numpy as np
from . import audio
from functools import partial
from concurrent.futures import ProcessPoolExecutor


###################
# WRITE META DATA #
###################
def write_meta_data(metadata, out_dir, meta_text, frame_shift_ms):
	with open(os.path.join(out_dir, meta_text), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
		frames = sum([m[2] for m in metadata])
		hours = frames * frame_shift_ms / (3600 * 1000)
		print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
		print('Max input length:  %d' % max(len(m[3]) for m in metadata))
		print('Max output length: %d' % max(m[2] for m in metadata))


###################
# BUILD FROM PATH #
###################
"""
	Preprocesses the dataset from given input paths into a given output directory.
	Use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
	can omit it and just call _process_utterance on each input if you want.

	Args:
		transcript_path: The path to the transcript file
		wav_dir: The directory where the audio is contained
		out_dir: The directory to write the output into
		num_workers: Optional number of worker processes to parallelize across
		tqdm: You can optionally pass tqdm to get a nice progress bar

	Returns:
		A list of tuples describing the training examples. This should be written to meta_text.txt by 'write_meta_data'
"""
def build_from_path(transcript_path, wav_dir, out_dir, file_suffix, num_workers=1, tqdm=lambda x: x):
	executor = ProcessPoolExecutor(max_workers=num_workers)
	futures = []
	index = 1
	with open(transcript_path, encoding='utf-8') as f:
		for line in f:
			tokens = line.strip().split('|')
			wav_path = os.path.join(wav_dir, '%s.%s' % (tokens[0], file_suffix))
			text = tokens[1]
			futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
			index += 1
	return [future.result() for future in tqdm(futures)]


#####################
# PROCESS UTTERANCE #
#####################
"""
	Preprocesses a single utterance audio/text pair.

	This writes the mel and linear scale spectrograms to disk and returns a tuple to write
	to the meta_text.txt file.

	Args:
		out_dir: The directory to write the spectrograms into
		index: The numeric index to use in the spectrogram filenames.
		wav_path: Path to the audio file containing the speech input
		text: The text spoken in the input audio file

	Returns:
		A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to meta_text.txt
"""
def _process_utterance(out_dir, index, wav_path, text):

	# Load the audio to a numpy array:
	wav = audio.load_wav(wav_path)

	# Compute the linear-scale spectrogram from the wav:
	spectrogram = audio.spectrogram(wav).astype(np.float32)
	n_frames = spectrogram.shape[1]

	# Compute a mel-scale spectrogram from the wav:
	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

	# Write the spectrograms to disk:
	spectrogram_filename = 'meta_spec_%05d.npy' % index
	mel_filename = 'meta_mel_%05d.npy' % index
	np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
	np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example:
	return (spectrogram_filename, mel_filename, n_frames, text)

