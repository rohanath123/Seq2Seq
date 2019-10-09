import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import re
import unicodedata

SOS_token  = 0
EOS_token = 1

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.index2word = {0: 'SOS', 1: 'EOS'}
		self.word2count = {}
		self.n = 2

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n
			self.index2word[self.n] = word
			self.n += 1
			self.word2count[word] = 1

		else:
			self.word2count[word] += 1


	def print_everything(self):
		print(self.word2index)
		print(self.word2count)
		print(self.index2word)
		print(self.n)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2):
	lines = open('D:/Deep Learning Training Data/Seq2Seq/fra-eng/fra.txt', encoding = 'utf-8').read().strip().split('\n')

	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

	input_lang = Lang(lang1)
	output_lang = Lang(lang2)

	return input_lang, output_lang, pairs

		
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ")

def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[0].startswith(eng_prefixes)

def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2):
	input_lang, output_lang, pairs = readLangs(lang1, lang2)
	
	pairs = filterPairs(pairs)
	for pair in pairs:
		input_lang.addSentence(pair[0])
		output_lang.addSentence(pair[1])

	return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra')

def indexesFromSentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]

def tensorsFromSentence(lang, sentence):
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype = torch.long).view(-1, 1)

def tensorsFromPair(p):
	input_tensor = tensorsFromSentence(input_lang, p[0])
	output_tensor = tensorsFromSentence(output_lang, p[1])
	return (input_tensor, output_tensor)
	