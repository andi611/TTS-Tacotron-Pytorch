# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ loss.py ]
#   Synopsis     [ Loss for the Tacotron model ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
from torch import nn


#################
# TACOTRON LOSS #
#################
class TacotronLoss(nn.Module):
	def __init__(self, sample_rate, linear_dim):
		super(TacotronLoss, self).__init__()
		self.sample_rate = sample_rate
		self.linear_dim = linear_dim
		self.criterion = nn.L1Loss()
		self.criterion_gate = nn.BCEWithLogitsLoss()

	def forward(self, model_output, targets):
		mel_outputs, mel = model_output[0], targets[0]
		linear_outputs, linear = model_output[1], targets[1]
		gate_outputs, gate = model_output[2], targets[2]

		mel_loss = cself.criterion(mel_outputs, mel)
		n_priority_freq = int(3000 / (self.sample_rate * 0.5) * self.linear_dim)
		linear_loss = 0.5 * self.criterion(linear_outputs, linear) + 0.5 * self.criterion(linear_outputs[:, :, :n_priority_freq], linear[:, :, :n_priority_freq])
		gate_loss = self.criterion_gate(gate_outputs, gate)
		
		loss = mel_loss + linear_loss + gate_loss
		losses = [loss, mel_loss, linear_loss, gate_loss]
		return losses


#########################
# GET MASK FROM LENGTHS #
#########################
"""
	Get mask tensor from list of length

	Args:
		memory: (batch, max_time, dim)
		memory_lengths: array like
"""
def get_rnn_mask_from_lengths(memory, memory_lengths):
	mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
	for idx, l in enumerate(memory_lengths):
		mask[idx][:l] = 1
	return ~mask

