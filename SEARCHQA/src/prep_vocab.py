import json
import argparse


def load_dict(fname):
	with open(fname, 'r') as fp:
		data = json.load(fp)
	return data


def dump_dict(fname, data):
	with open(fname, 'w') as fp:
		json.dump(data, fp)


def prep_vocab(tok_data_list, glove_word_list):
	glove_word_set = set(glove_word_list)
	vocab_words = set()
	char_vocab = set()
	for tok_data in tok_data_list:
		for p in tok_data:
			vocab_words.update(p['context.tokens'])
			char_vocab.update(p['context'])
			for qa in p['qas']:
				vocab_words.update(qa['question.tokens'])
				char_vocab.update(qa['question'])
	common_words = vocab_words.intersection(glove_word_set)

	return list(common_words), list(char_vocab)


def get_words_glove(glovefname):
	with open(glovefname, 'r') as fp:
		lines = fp.readlines()
	words = [line.split()[0] for line in lines]
	return words


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Prepare vocabulary from tokenized train and valid data')
	parser.add_argument('-toktr', '--tokenized_train', type=str, default='data/tokenized-train.json',
			    help='tokenized training file path')
	parser.add_argument('-tokval', '--tokenized_val', type=str, default='data/tokenized-val.json',
			    help='tokenized valid file path')
	parser.add_argument('-id2w', '--id2wordfname', type=str, default='prep-data/id2word.json',
			    help='id2word file path for dumping')
	parser.add_argument('-id2c', '--id2charfname', type=str, default='prep-data/id2char.json',
			    help='id2char file path for dumping')
	parser.add_argument('-glovef', '--glovefname', type=str, default='glove.840B.300d.txt',
			    help='glove file name')
	args = parser.parse_args()

	glove_word_list = get_words_glove(args.glovefname)

	tok_tr = load_dict(args.tokenized_train)
	tok_val = load_dict(args.tokenized_val)
	id2word, id2char = prep_vocab([tok_tr, tok_val], glove_word_list)
	dump_dict(args.id2wordfname, id2word)
	dump_dict(args.id2charfname, id2char)
