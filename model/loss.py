from torch import nn

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

		mel_loss = criterions[0](mel_outputs, mel)
		n_priority_freq = int(3000 / (self.sample_rate * 0.5) * self.linear_dim)
		linear_loss = 0.5 * self.criterion(linear_outputs, linear) + 0.5 * self.criterion(linear_outputs[:, :, :n_priority_freq], linear[:, :, :n_priority_freq])
		gate_loss = self.criterion_gate(gate_outputs, gate)
		
		loss = mel_loss + linear_loss + gate_loss
		losses = [loss, mel_loss, linear_loss, gate_loss]
		return losses