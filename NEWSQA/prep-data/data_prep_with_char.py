import sys
import json
import argparse
from tqdm import tqdm

class DataPrep(object):
	def __init__(self, datafile, id2word_vocab_file,
		     id2char_vocab_file,
		     maxchar=10, **kwargs):
		with open(id2word_vocab_file, 'r') as fp:
			vocab_words = json.load(fp)
		with open(id2char_vocab_file, 'r') as fp:
			vocab_chars = json.load(fp)
		with open(datafile, 'r') as fp:
			paragraphs = json.load(fp)

		self.paragraphs = paragraphs
		self.vocab = ["<pad>", "<unk>"] + vocab_words
		self.vocab_size = len(self.vocab)
		self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}
		self.char_vocab = ["<padc>", "<unkc>"] + vocab_chars
		self.char_vocab_size = len(self.char_vocab)
		self.char_reverse_vocab = {c: idx for idx, c in enumerate(self.char_vocab)}
		self.maxwordlen = maxchar
		self.wh_words = ['what', 'who', 'how', 'when', 'which', 'where', 'why']
	
	def get_word_id(self, w):
		if w in self.reverse_vocab:
			return self.reverse_vocab[w]
		else:
			return self.reverse_vocab['<unk>']

	def get_id_list(self, wordlist):
		return [self.get_word_id(w) for w in wordlist]

	def get_char_id(self, ch):
		if ch in self.char_reverse_vocab:
			return self.char_reverse_vocab[ch]
		else:
			return self.char_reverse_vocab['<unkc>']

	def get_charidlist_for_word(self, w):
		if len(w) <= self.maxwordlen:
			return [self.get_char_id(ch) for ch in w] + [self.char_reverse_vocab['<padc>']]*(self.maxwordlen - len(w))
		else:
			return [self.get_char_id(ch) for ch in w[:self.maxwordlen]]

	def get_char_id_list(self, wordlist):
		return [self.get_charidlist_for_word(w) for w in wordlist]

	def qwh_onehot_vec(self, qwords):
		first_idx = -1
		wh_one_hot_vec1 = [0] * len(qwords)
		wh_one_hot_vec2 = [0] * len(qwords)
		for word_idx in range(len(qwords)):
			if qwords[word_idx].lower() in self.wh_words:
				first_idx = word_idx
				break
		if first_idx == -1:
			first_idx = 0
		if len(qwords) < 2:
			assert first_idx == 0
			second_idx = 0
		elif first_idx == len(qwords) - 1:
			second_idx = first_idx - 1
		else:
			second_idx = first_idx + 1
		wh_one_hot_vec1[first_idx] = 1
		wh_one_hot_vec2[second_idx] = 1
		return wh_one_hot_vec1, wh_one_hot_vec2

	def prep_data(self):
		cqa = {}
		for p in tqdm(self.paragraphs):
			ctx_words = p['context.tokens']
			ctx_idxs = self.get_id_list(ctx_words)
			ctx_char_idxs = self.get_char_id_list(ctx_words)
			for qa in p['qas']:
				qid = qa['id']
				q_words = qa['question.tokens']
				q_idxs = self.get_id_list(q_words)
				q_char_idxs = self.get_char_id_list(q_words)
				q_wh, q_wh_next = self.qwh_onehot_vec(q_words)
				label_start = [0]*len(ctx_words)
				label_start[qa['answers'][0]['answer_start']] = 1
				label_length = len(qa['answers'][0]['text.tokens'])
			
				data_dict = {'context': ctx_idxs, 'context_chars': ctx_char_idxs, 'question': q_idxs, 'question_chars': q_char_idxs,
                                            'q_wh': q_wh, 'q_wh_next': q_wh_next,
                                            'answer_start': label_start, 'answer_string': label_length}
				cqa[qid] = data_dict
		return cqa


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train and Test data preparation for SQuAD from the tokenized data')
	parser.add_argument('-data', '--datafile', type=str, default='train-v1.1.withuni.case.json',
				help = 'tokenized data json file name')

	parser.add_argument('-id2w', '--id2wordf', type=str, default='id2word.json',
				help='id2word vocab file name')

	parser.add_argument('-id2c', '--id2charf', type=str, default='id2char.json',
                                help='id2char vocab file name')

	parser.add_argument('-wr', '--writef', type=str, default='data-wo-pad/with-char-idx/train_withuni.json',
				help='write file name')
	args = parser.parse_args()

	data_file = args.datafile
	vocab_file = args.id2wordf
	char_vocab_file = args.id2charf
	writefile = args.writef

	squad = DataPrep(data_file, vocab_file, char_vocab_file)
	cqa = squad.prep_data()
	print "Dumping..."
	with open(writefile, 'w') as fw:
		json.dump(cqa, fw)
