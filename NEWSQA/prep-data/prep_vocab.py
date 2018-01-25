import json
import argparse

def load_dict(fname):
	with open(fname, 'r') as fp:
		data = json.load(fp)
	return data


def dump_dict(fname, data):
	with open(fname, 'w') as fp:
		json.dump(data, fp)


def prep_vocab(tokenized_data):
	vocab_words = set()
	for para in tokenized_data:
		vocab_words.update(para['context.tokens'])
		for qa in para['qas']:
			vocab_words.update(qa['question.tokens'])
	return list(vocab_words)


def prep_char_vocab(tokenized_data):
	char_vocab = set()
	for para in tokenized_data:
		char_vocab.update(para['context'])
		for qa in para['qas']:
			char_vocab.update(qa['question'])
	return list(char_vocab)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Vocabulary preparation for SQuAD')
	parser.add_argument('-trtokdata', '--traintokdatafile', type=str, default='data/tokenized-train-v1.1.json',
                                help = 'tokenized train data json file name')
	parser.add_argument('-devtokdata', '--devtokdatafile', type=str, default='data/tokenized-dev-v1.1.json',
                                help = 'tokenized dev data json file name')

        parser.add_argument('-testtokdata', '--testtokdatafile', type=str, default='data/tokenized-test-v1.1.json',
                                help = 'tokenized dev data json file name')

	parser.add_argument('-id2w', '--id2wordf', type=str, default='id2word.json',
                                help='id2word vocab file name')

	parser.add_argument('-id2c', '--id2charf', type=str, default='id2char.json',
                                help='id2char vocab file name')

	args = parser.parse_args()

	train_tok_data = load_dict(args.traintokdatafile)
	dev_tok_data = load_dict(args.devtokdatafile)
        test_tok_data = load_dict(args.testtokdatafile)
	word_vocab_file = args.id2wordf
	char_vocab_file = args.id2charf
	
	word_vocab = set()
	word_vocab.update(prep_vocab(train_tok_data))
	word_vocab.update(prep_vocab(dev_tok_data))
        word_vocab.update(prep_vocab(test_tok_data))
	word_vocab = list(word_vocab)
	dump_dict(word_vocab_file, word_vocab)

	char_vocab = set()
	char_vocab.update(prep_char_vocab(train_tok_data))
	char_vocab.update(prep_char_vocab(train_tok_data))
        char_vocab.update(prep_char_vocab(test_tok_data))
	char_vocab = list(char_vocab)
	dump_dict(char_vocab_file, char_vocab)
