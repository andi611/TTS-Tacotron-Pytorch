# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ test.py ]
#   Synopsis     [ Testing algorithms for a trained Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import sys
import nltk
import argparse
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
#--------------------------------#
import torch
from torch.autograd import Variable
#--------------------------------#
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize, plot_alignment
#--------------------------------#
from model.tacotron import Tacotron
from config import config, get_test_args


############
# CONSTANT #
############
USE_CUDA = torch.cuda.is_available()


##################
# TEXT TO SPEECH #
##################
def tts(model, text):
	"""Convert text to speech waveform given a Tacotron model.
	"""
	if USE_CUDA:
		model = model.cuda()
	
	# NOTE: dropout in the decoder should be activated for generalization!
	# model.decoder.eval()
	model.encoder.eval()
	model.postnet.eval()

	sequence = np.array(text_to_sequence(text))
	sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
	if USE_CUDA:
		sequence = sequence.cuda()

	# Greedy decoding
	mel_outputs, linear_outputs, gate_outputs, alignments = model(sequence)

	linear_output = linear_outputs[0].cpu().data.numpy()
	spectrogram = audio._denormalize(linear_output)
	alignment = alignments[0].cpu().data.numpy()

	# Predicted audio signal
	waveform = audio.inv_spectrogram(linear_output.T)

	return waveform, alignment, spectrogram


####################
# SYNTHESIS SPEECH #
####################
def synthesis_speech(model, text, figures=True, path=None):
	waveform, alignment, spectrogram = tts(model, text)
	if figures:
		test_visualize(alignment, spectrogram, path)
	librosa.output.write_wav(path + '.wav', waveform, config.sample_rate)


########
# MAIN #
########
def main():

	#---initialize---#
	args = get_test_args()

	model = Tacotron(n_vocab=len(symbols),
					 embedding_dim=config.embedding_dim,
					 mel_dim=config.num_mels,
					 linear_dim=config.num_freq,
					 r=config.outputs_per_step,
					 padding_idx=config.padding_idx,
					 attention=config.attention,
					 use_mask=config.use_mask)

	#---handle path---#
	checkpoint_path = os.path.join(args.ckpt_dir, args.checkpoint_name + args.model_name + '.pth')
	os.makedirs(args.result_dir, exist_ok=True)
	
	#---load and set model---#
	print('Loading model: ', checkpoint_path)
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint["state_dict"])
	model.decoder.max_decoder_steps = config.max_decoder_steps # Set large max_decoder steps to handle long sentence outputs
		
	if args.interactive == True:
		output_name = args.result_dir + args.model_name

		#---testing loop---#
		while True:
			try:
				text = str(input('< Tacotron > Text to speech: '))
				print('Model input: ', text)
				synthesis_speech(model, text=text, figures=args.plot, path=output_name)
			except KeyboardInterrupt:
				print()
				print('Terminating!')
				break

	elif args.interactive == False:
		output_name = args.result_dir + args.model_name + '/'
		os.makedirs(output_name, exist_ok=True)

		#---testing flow---#
		with open(args.test_file_path, 'r', encoding='utf-8') as f:
			
			lines = f.readlines()
			for idx, line in enumerate(lines):
				print("{}: {} - ({} chars)".format(idx+1, line, len(line)))
				synthesis_speech(model, text=line, figures=args.plot, path=output_name+str(idx+1))

		print("Finished! Check out {} for generated audio samples.".format(output_name))
	
	else:
		raise RuntimeError('Invalid mode!!!')
		
	sys.exit(0)

if __name__ == "__main__":
	main()
