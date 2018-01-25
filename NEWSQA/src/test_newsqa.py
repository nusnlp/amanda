import sys
import os
sys.path.append('../amanda/')

import json
import numpy
from numpy import array
import h5py
import random
from random import shuffle
from tqdm import tqdm
import argparse

from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers import Layer, recurrent, Input, merge
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop, Adam, Adamax, Adadelta

from model import QAModel
import newsqa_evaluator as Evaluator


def compute_mask(self, input, input_mask=None):
	# to override Keras Layer's member function
	if not hasattr(self, 'supports_masking') or not self.supports_masking:
		if input_mask is not None:
			if type(input_mask) is list:
				if any(input_mask):
					warnings.warn(
						'Layer ' + self.name + ' does not support masking, mask info will not be passed on')
				else:
					warnings.warn(
						'Layer ' + self.name + ' does not support masking, mask info will not be passed on')
	return None

Layer.compute_mask=compute_mask


def load_data(fname):
	with open(fname, 'r') as fp:
		data = json.load(fp)

	return data


def load_paras(fname):
	with open(fname, 'r') as fp:
		paras = json.load(fp)
	return paras


def get_ans_string_single_post_pad_search_updated(context, context_words,
						  ans_start_pred, ans_end_pred,
						  maxspan=15):
	ans_start = ans_start_pred[:len(context_words)]
	ans_end = ans_end_pred[:len(context_words)]
	p = numpy.zeros((len(context_words), len(context_words)))
	for i in range(len(context_words)):
		for j in range(i, min(i + maxspan, len(context_words))):
			p[i, j] = ans_start[i] * ans_end[j]
	loc = numpy.argmax(p)
	start_ind = int(loc / len(context_words))
	end_ind = loc - start_ind * len(context_words)
	indices = [start_ind, end_ind]

	context = context.replace("``", '"').replace("''", '"')
	char_idx = 0
	char_start, char_stop = None, None
	for word_idx, word in enumerate(context_words):
		word = word.replace("``", '"').replace("''", '"')
		# print word
		char_idx = context.find(word, char_idx)
		assert char_idx >= 0
		if word_idx == indices[0]:
			char_start = char_idx
		char_idx += len(word)
		if word_idx == indices[1]:
			char_stop = char_idx
			break

	assert char_start is not None
	assert char_stop is not None

	return context[char_start:char_stop]


def get_prediction_answer_strings_updated(tokenized_datapath, textdatapath, dev_pred_dict):
	textdata = load_data(textdatapath)['data']
	ctx_dict = {}
	for a in textdata:
		for p in a['paragraphs']:
			for qa in p['qas']:
				ctx_dict[qa['id']] = p['context']
	paragraphs = load_paras(tokenized_datapath)
	keys_list = dev_pred_dict.keys()
	pred_dict = {}
	for paragraph in paragraphs:
		context_words = paragraph['context.tokens']
		for qa in paragraph['qas']:
			# print qa['id']
			# print ctx_dict[qa['id']]
			if qa['id'] in keys_list:
				ans_start_pred = dev_pred_dict[qa['id']]['ans_start_pred']
				ans_end_pred = dev_pred_dict[qa['id']]['ans_end_pred']
				pred_dict[qa['id']] = get_ans_string_single_post_pad_search_updated(ctx_dict[qa['id']],
												    context_words,
												    ans_start_pred,
												    ans_end_pred)
	return pred_dict


def predict_batchwise(data, model, batch_size=4, maxwordlen=10):
	qids = data.keys()
	num_samples = len(qids)
	num_batches = num_samples / batch_size
	pred_dict = {}
	if num_batches * batch_size == num_samples:
		max_round = num_batches
	else:
		max_round = num_batches + 1
	for indx in tqdm(range(0, max_round)):
		S, Q = [], []
		Qwh, Qwh_next = [], []
		story_lengths, query_lengths = [], []
		uper_bound = min(num_samples, (indx + 1) * batch_size)
		batch_qids = qids[indx * batch_size: uper_bound]
		keylist = qids[indx * batch_size: uper_bound]
		for key in keylist:
			story = data[key]['context']
			q = data[key]['question']
			qwh = data[key]['q_wh']
			qwh_next = data[key]['q_wh_next']
			story_lengths.append(len(story))
			query_lengths.append(len(q))
			S.append(story)
			Q.append(q)
			Qwh.append(qwh)
			Qwh_next.append(qwh_next)

		max_storylen = max(story_lengths)
		max_querylen = max(query_lengths)
		S = pad_sequences(S, maxlen=max_storylen, padding='post')
		Q = pad_sequences(Q, maxlen=max_querylen, padding='post')
		Sch = [data[key]['context_chars'] + [[0] * maxwordlen] * (
		max_storylen - len(data[key]['context_chars'])) for key in keylist]
		Qch = [data[key]['question_chars'] + [[0] * maxwordlen] * (
		max_querylen - len(data[key]['question_chars'])) for key in keylist]

		Qwh = pad_sequences(Qwh, maxlen=max_querylen, padding='post')
		Qwh_next = pad_sequences(Qwh_next, maxlen=max_querylen, padding='post')

		answer = model.predict(
			{"story": S, "question": Q, "story_char": array(Sch), "question_char": array(Qch),
			 "q_wh": Qwh, "q_wh_next": Qwh_next},
			batch_size=batch_size)

		for i in range(len(batch_qids)):
			pred_dict[batch_qids[i]] = {}
			pred_dict[batch_qids[i]]['ans_start_pred'] = answer[i][0]
			pred_dict[batch_qids[i]]['ans_end_pred'] = answer[i][1]

	return pred_dict


def listify_pred_dict(pred_dict):
	new_pred = {}
	for key in pred_dict.keys():
		new_pred[key] = {}
		new_pred[key]['ans_start_pred'] = \
			pred_dict[key]['ans_start_pred'].tolist()
		new_pred[key]['ans_end_pred'] = \
			pred_dict[key]['ans_end_pred'].tolist()
	return new_pred


if __name__ == "__main__":
	#
	parser = argparse.ArgumentParser(description='Testing on NewsQA dataset')
	parser.add_argument('-w', '--weightpath', type=str, default=None,
			    	help='path of the model weight')
	parser.add_argument('-word_emb', '--embed_mat_path', type=str,
			    default='prep-data/embed_mat.npy', help='path of the word embed mat')
	parser.add_argument('-char_emb', '--char_embedding', type=bool, default=True,
			    help='whether to consider char embedding')
	parser.add_argument('-ch_embdim', '--char_embed_dim', type=int, default=50,
			    help='character embedding dimension')
	parser.add_argument('-maxwlen', '--maxwordlen', type=int, default=10,
			    help='maximum number of chars in a word')
	parser.add_argument('-chfw', '--char_cnn_filter_width', type=int, default=5,
			    help='Character level CNN filter width')
	parser.add_argument('-bm', '--border_mode', type=str, default=None,
			    help='border mode for char CNN')
	parser.add_argument('-qt', '--qtype', type=str, default='wh2',
			    help='type of the question type representation')
	parser.add_argument('-hdim', '--hidden_dim', type=int, default=150)
	parser.add_argument('-rnnt', '--rnn_type', type=str, default='lstm',
			    help='Type of the building block RNNs')
	parser.add_argument('-dop', '--dropout_rate', type=float, default=0.3,
			    help='Dropout rate')
	parser.add_argument('-istr', '--is_training', type=bool, default=False,
			    help='Is it a training script?')
	parser.add_argument('-pbs', '--predict_batch_size', type=int, default=4,
			    help='Prediction batch size')

	# data inputs
	parser.add_argument('-testjs', '--test_json', type=str, default='data/test-v1.1.json',
			    help='formatted test JSON file')
	parser.add_argument('-toktest', '--tok_test_json', type=str,
			    default='data/tokenized-test-v1.1.json',
			    help='tokenized test JSON file')
	parser.add_argument('-indextest', '--indexed_test_json', type=str,
			    default='data/test_indexed.json',
			    help='Indexed test JSON file')


	parser.add_argument('-id2c', '--id2char', type=str, default='prep-data/id2char.json',
			    help='id2char JSON file')

	args = parser.parse_args()

	char_vocab_size = len(load_data(args.id2char)) + 2
	char_embed_dim = args.char_embed_dim

	args.char_vocab_size = char_vocab_size
	args.embed_mat = numpy.load(args.embed_mat_path)
	learning_rate = 0.001

	if True:
		G = QAModel(args)
		model = G.create_model_graph()

		opt = Adam(lr=learning_rate, clipnorm=5.0)
		model.compile(optimizer=opt,
			      loss='categorical_crossentropy', metrics=['accuracy'])

		model.load_weights(args.weightpath)
		print "Model loaded..."

		##Testing
		print "Loading data..."
		textdatapath = args.test_json
		processed_data = args.tok_test_json
		datafile = args.indexed_test_json
		dev_data = load_data(datafile)

		print "Preparing prediction dictionary..."
		dev_pred_dict = predict_batchwise(dev_data, model,
						  batch_size=args.predict_batch_size,
						  maxwordlen=args.maxwordlen)

		predictions = get_prediction_answer_strings_updated(processed_data,
								    textdatapath,
								    dev_pred_dict)
		
		print "Extracting ground truth answers..."
		dataset = load_data(textdatapath)['data']

		print "Calculating stats..."
		stats = Evaluator.evaluate(dataset, predictions)
		print(json.dumps(stats))
		print "--------------------------------------------------------------------\n"

