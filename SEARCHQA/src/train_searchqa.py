import sys
sys.path.append('../amanda/')
import os
import json
import numpy
numpy.random.seed(1337)
from numpy import array
import h5py
import random
from random import shuffle
from tqdm import tqdm
import argparse
import logging
import shutil

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
import searchqa_evaluator as Evaluator


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


def get_ans_string_single_post_pad_search_updated(context,
												  context_words,
												  ans_start_pred,
												  ans_end_pred,
												  maxctxlen, maxspan):
	# answer span
	context_words = context_words[:min(len(context_words), maxctxlen)]
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
		# assert char_idx >= 0
		if word_idx == indices[0]:
			char_start = char_idx
		char_idx += len(word)
		if word_idx == indices[1]:
			char_stop = char_idx
			break

	if char_start is None or char_stop is None:
		return ' '.join(context_words[indices[0]:indices[1] + 1])
	assert char_start is not None
	assert char_stop is not None

	return context[char_start:char_stop]


def get_prediction_answer_strings_updated(tokenized_datapath,
										  textdatapath, dev_pred_dict,
										  maxctxlen=200, maxspan=1):
	textdata = load_data(textdatapath)['data']
	ctx_dict = {}
	for a in textdata:
		for p in a['paragraphs']:
			for qa in p['qas']:
				ctx_dict[qa['id']] = p['context']
	paragraphs = load_paras(tokenized_datapath)
	keys_list = dev_pred_dict.keys()
	pred_dict = {}
	for paragraph in tqdm(paragraphs):
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
																					ans_end_pred, maxctxlen, maxspan)
	return pred_dict


def minibatch_generator(data, batch_size=60, maxwordlen=10, shuffling=True):
	qids = data.keys()
	if shuffling:
		shuffle(qids)
	num_samples = len(qids)
	num_batches = num_samples / batch_size
	if num_batches*batch_size == num_samples:
		max_round = num_batches
	else:
		max_round = num_batches + 1
	while 1:
		for indx in range(0, max_round):
			S, Q, Astart, Aend = [], [], [], []
			# Sch, Qch = [], []
			# astart_end = []
			story_lengths, query_lengths = [], []
			uper_bound = min(num_samples, (indx + 1) * batch_size)

			keylist = qids[indx * batch_size: uper_bound]
			for key in keylist:
				story = data[key]['context']
				q = data[key]['question']
				astart = data[key]['answer_start']
				alen = data[key]['answer_string']
				assert len(story) == len(astart)
				start_ind = astart.index(1)
				ans_len = alen
				aend = [0] * len(astart)
				assert start_ind + ans_len - 1 < len(astart)
				aend[start_ind + ans_len - 1] = 1
				story_lengths.append(len(story))
				query_lengths.append(len(q))
				S.append(story)
				Q.append(q)
				Astart.append(astart)
				Aend.append(aend)

			max_storylen = max(story_lengths)
			max_querylen = max(query_lengths)
			Astart = pad_sequences(Astart, maxlen=max_storylen, padding='post')
			Aend = pad_sequences(Aend, maxlen=max_storylen, padding='post')
			Astart = Astart[:, numpy.newaxis, :]
			Aend = Aend[:, numpy.newaxis, :]
			astart_end = numpy.concatenate((Astart, Aend), axis=1)

			Sch = [data[key]['context_chars'] + [[0] * maxwordlen] * (
			max_storylen - len(data[key]['context_chars'])) for key in keylist]
			Qch = [data[key]['question_chars'] + [[0] * maxwordlen] * (
			max_querylen - len(data[key]['question_chars'])) for key in keylist]

			yield ({"story": pad_sequences(S, maxlen=max_storylen, padding='post'),
				"question": pad_sequences(Q, maxlen=max_querylen, padding='post'),
				"story_char": array(Sch), "question_char": array(Qch)},
				   {"ans_start_end": astart_end})


def predict_batchwise(data, model, batch_size=60,
					  maxwordlen=10, maxctxlen=200):
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
			story_lengths, query_lengths = [], []
			uper_bound = min(num_samples, (indx + 1) * batch_size)
			batch_qids = qids[indx * batch_size: uper_bound]
			keylist = qids[indx * batch_size: uper_bound]
			for key in keylist:
					story = data[key]['context']
					q = data[key]['question']
					story_lengths.append(len(story))
					query_lengths.append(len(q))
					S.append(story)
					Q.append(q)

			max_storylen = min(max(story_lengths), maxctxlen)
			max_querylen = max(query_lengths)
			S = pad_sequences(S, maxlen=max_storylen, padding='post', truncating='post')
			Q = pad_sequences(Q, maxlen=max_querylen, padding='post')
			Sch = [data[key]['context_chars'] + [[0] * maxwordlen] * (max_storylen - len(data[key]['context_chars']))
				   if len(data[key]['context_chars']) <= maxctxlen else data[key]['context_chars'][:maxctxlen]
				   for key in keylist]
			Qch = [data[key]['question_chars'] + [[0] * maxwordlen] * (
					max_querylen - len(data[key]['question_chars'])) for key in keylist]


			answer = model.predict({"story": S, "question": Q,
									"story_char": array(Sch),
									"question_char": array(Qch)},
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
		new_pred[key]['ans_start_pred'] = pred_dict[key]['ans_start_pred'].tolist()
		new_pred[key]['ans_end_pred'] = pred_dict[key]['ans_end_pred'].tolist()
	return new_pred


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Training on SearchQA dataset')
	# data inputs
	parser.add_argument('-indextr', '--indexed_train_json', type=str,
						default='data/train_indexed_sample.json',
						help='Indexed train JSON file')

	parser.add_argument('-devjs', '--dev_json', type=str, default='data/val.json',
						help='formatted dev JSON file')
	parser.add_argument('-tokdev', '--tok_dev_json', type=str,
						default='data/tokenized-dev.json',
						help='tokenized dev JSON file')
	parser.add_argument('-indexdev', '--indexed_dev_json', type=str,
						default='data/dev_indexed.json',
						help='Indexed dev JSON file')

	parser.add_argument('-id2c', '--id2char', type=str, default='prep-data/id2char.json',
						help='id2char JSON file')

	# model configs
	parser.add_argument('-pretrained', '--pretrained_weightpath', type=str,
						default=None,
						help='path of any pretrained model weight')
	parser.add_argument('-initepoch', '--initial_epoch', type=int, default=0,
						help='Initial epoch count')
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
						help='learning rate')
	parser.add_argument('-clip', '--clipnorm', type=float, default=5.0,
						help='Clipnorm threshold')
	parser.add_argument('-optim', '--optimizer', type=str, default='adam',
						choices=['adamax', 'adam', 'rmsprop'],
						help='backpropagation algorithm')

	parser.add_argument('-word_emb', '--embed_mat_path', type=str,
						default='prep-data/embed_mat.npy',
						help='path of the word embed mat (.npy format)')
	parser.add_argument('-embtr', '--embed_trainable', type=bool, default=False,
						help='whether to refine word embedding weights during training')
	parser.add_argument('-char_emb', '--char_embedding', type=bool, default=True,
						help='whether to consider char embedding')
	parser.add_argument('-ch_embdim', '--char_embed_dim', type=int, default=50,
						help='character embedding dimension')
	parser.add_argument('-maxwlen', '--maxwordlen', type=int, default=10,
						help='maximum number of chars in a word')
	parser.add_argument('-chfw', '--char_cnn_filter_width', type=int, default=5,
						help='Character level CNN filter width')
	parser.add_argument('-bm', '--border_mode', type=str, default='same',
						help='border mode for char CNN')

	parser.add_argument('-qt', '--qtype', type=str, default='first2',
						help='type of the question type representation')
	parser.add_argument('-hdim', '--hidden_dim', type=int, default=150)
	parser.add_argument('-rnnt', '--rnn_type', type=str, default='lstm',
						help='Type of the building block RNNs')
	parser.add_argument('-dop', '--dropout_rate', type=float, default=0.3,
						help='Dropout rate')
	parser.add_argument('-istr', '--is_training', type=bool, default=True,
						help='Is it a training script?')
	parser.add_argument('-ne', '--num_epoch', type=int, default=10,
						help='number of training epochs')
	parser.add_argument('-trbs', '--training_batch_size', type=int, default=60,
						help='Training batch size')
	parser.add_argument('-adptlr', '--adapt_lr', type=bool, default=True,
						help='whether to reduce learning rate')
	parser.add_argument('-is_val', '--is_valid', type=bool, default=False,
						help='validate on the dev data?')
	parser.add_argument('-pbs', '--predict_batch_size', type=int, default=60,
						help='Prediction batch size')

	# experiment directory
	parser.add_argument('-exp', '--baseexp', type=str, default='exp-searchqa',
						help='name of the experiment directory')

	args = parser.parse_args()
	# -----------------------------------------------------------------------------#

	char_vocab_size = len(load_data(args.id2char)) + 2  # add 2 for pad and unk
	char_embed_dim = args.char_embed_dim

	baseexp = args.baseexp
	if not os.path.isdir(baseexp):
		os.makedirs(baseexp)

	logdir = baseexp + '/log'
	if not os.path.isdir(logdir):
		os.makedirs(logdir)
	logfile = logdir + '/logfile'
	if os.path.exists(logfile):
		with open(logfile, 'w') as fw:
			fw.write('')
	logging.basicConfig(filename=logfile,
						format='%(levelname)s:%(message)s',
						level=logging.INFO)

	logging.info('***Arguments***')
	logging.info(args)
	logging.info('-' * 100)

	logging.info("Loading embedding matrix..")
	embed_mat = numpy.load(args.embed_mat_path)
	args.embed_mat = embed_mat
	args.char_vocab_size = char_vocab_size
	# -----------------------------------------------------------------------------#

	#learning_rate = 0.001

	logging.info("Training data loading...")
	train_data = load_data(args.indexed_train_json)

	if args.is_valid:
		logging.info("Validation data loading...")
		textdatapath = args.dev_json
		processed_data = args.tok_dev_json
		datafile = args.indexed_dev_json
		dev_data = load_data(datafile)
		dataset = load_data(textdatapath)['data']

		if os.path.isfile(baseexp + '/results.txt'):
			mode = 'a'
		else:
			mode = 'w'
		with open(baseexp + '/results.txt', mode) as fp:
			fp.write("######RESULTS######\n")

	# Initializations for tracking the best model
	prev_best_em = 0.0
	prev_best_f1 = 0.0
	prev_best_epoch = args.initial_epoch - 1

	logging.info('=' * 100)
	for epoch in range(args.initial_epoch, args.num_epoch):
		logging.info("Epoch: %s", str(epoch))
		G = QAModel(args)
		model = G.create_model_graph()

		logging.info("Compiling model..")
		# print "Learning rate:", args.learning_rate
		model = G.compile_model(model)
		logging.info("Model compiled..")

		exp = baseexp + "/epoch" + str(epoch)
		if not os.path.isdir(exp):
			os.makedirs(exp)

		if args.pretrained_weightpath is not None and epoch == args.initial_epoch:
			logging.info("Loading a pretrained weight")
			model.load_weights(args.pretrained_weightpath)
			if args.is_valid:
				logging.info("Evaluating the pretrained model")
				prev_dev_pred_dict = predict_batchwise(dev_data, model)
				prev_predictions = get_prediction_answer_strings_updated(
					processed_data, textdatapath, prev_dev_pred_dict)

				logging.info("Calculating stats for pretrained model..")
				prev_stats = Evaluator.evaluate(dataset, prev_predictions)
				prev_best_em = prev_stats['exact_match']
				prev_best_f1 = prev_stats['f1']
				print(json.dumps(prev_stats))
		elif prev_best_epoch != -1:
			logging.info("Loading the weights from epoch: %s",
				     str(prev_best_epoch))
			weight_dir = baseexp + "/epoch" + str(prev_best_epoch)
			files = os.listdir(weight_dir)
			weightfname = None
			for f in files:
				if "weights" in f:
					weightfname = f
					break
			assert weightfname is not None
			prev_best_weight_path = os.path.join(weight_dir, weightfname)
			model.load_weights(prev_best_weight_path)

		if epoch > -1:
			checkpointer = ModelCheckpoint(filepath=os.path.join(exp,
					"weights.{epoch:02d}-{loss:.2f}-{acc:.4f}.hdf5"),
					save_best_only=False)
			logging.info("Starting training...")

			model.fit_generator(minibatch_generator(train_data,
													batch_size=args.training_batch_size,
													maxwordlen=args.maxwordlen
													),
								samples_per_epoch=len(train_data),
								nb_epoch=1,
								callbacks=[checkpointer], max_q_size=10)
			logging.info("Training Finished..")


		if args.is_valid:
			##Testing
			logging.info("Preparing prediction dictionary...")
			dev_pred_dict = predict_batchwise(dev_data, model,
											  batch_size=args.predict_batch_size,
											  maxwordlen=args.maxwordlen
											  )
			predictions = get_prediction_answer_strings_updated(
				processed_data, textdatapath, dev_pred_dict)

			logging.info("Calculating stats...")
			stats = Evaluator.evaluate(dataset, predictions)
			logging.info(json.dumps(stats))
			print "-" * 100
			logging.info("-" * 100)
			with open(baseexp + '/results.txt', 'a') as fp:
				fp.write("###### Epoch: " + str(epoch) + " ######")
				fp.write("\n")
				fp.write("Learning rate: " + str(args.learning_rate) + '\n')
				json.dump(stats, fp)
				fp.write("\n")
				fp.write("-" * 100 + '\n')

			if stats['exact_match'] >= prev_best_em:
				pred_based_em = dev_pred_dict.copy()
				prev_best_em = stats['exact_match']
			if stats['f1'] >= prev_best_f1:
				pred_based_f1 = dev_pred_dict.copy()
				prev_best_f1 = stats['f1']
				prev_best_epoch = epoch
				weight_dir = baseexp + "/epoch" + str(epoch)
				files = os.listdir(weight_dir)
				weightfname = None
				for f in files:
					if "weights" in f:
						weightfname = f
						break
				assert weightfname is not None
				prev_best_weight_path = os.path.join(weight_dir, weightfname)
				shutil.copy2(prev_best_weight_path,
							 os.path.join(baseexp, 'best_model.hdf5'))
			else:
				if args.adapt_lr:
					args.learning_rate /= 2.0
		print "=" * 100
		logging.info("=" * 100)


	logging.info("=" * 100)
	logging.info("Finished.")
	print "Finished"
