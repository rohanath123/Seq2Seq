import torch 
import numpy as np 
from seq2seq import *
from data_cleaning import *
import random
from torch import optim


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length = 10):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

	loss = 0

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
		encoder_outputs[ei] = encoder_hidden[0, 0]

	decoder_input = torch.tensor([[SOS_token]])
	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)

	if use_teacher_forcing:
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			loss+= criterion(decoder_output, target_tensor[di])
			
			decoder_input = target_tensor[di]

	else:
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()

			loss += criterion(decoder_output, target_tensor[di])
			

			if decoder_input.item() == EOS_token:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item()/target_length

def trainIters(encoder, decoder, iters, learning_rate = 0.01):
	print_loss_total = []

	encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)

	train_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(iters)]
	criterion = nn.NLLLoss()

	for iter in range(iters):
		train_pair = train_pairs[iter]
		input_tensor = train_pair[0]
		target_tensor = train_pair[1]

		loss= train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

		print("Iter = ", iter, " Loss = ", loss)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n, dropout_p=0.1)
trainIters(encoder1, attn_decoder1, 75000)
		

	





