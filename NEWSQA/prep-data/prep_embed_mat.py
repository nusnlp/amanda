import json
import numpy
import argparse


def loadGlovemodel(glove_file):
	model = {}
	with open(glove_file, 'r') as f:
		for line in f:
			splitLine = line.split()
			word = splitLine[0]
			embedding = [float(val) for val in splitLine[1:]]
			model[word] = embedding
	return model


def get_embed_mat(glove_file, id2word_file):
	model = loadGlovemodel(glove_file)
	d = len(model[model.keys()[0]])
	count = 0
	with open(id2word_file, 'r') as fp:
		vocab_words = json.load(fp)
	vocab_words = ["<pad>", "<unk>"] + vocab_words
	embed_mat = numpy.zeros((len(vocab_words), d))
	for idx, word in enumerate(vocab_words):
		word_vec = model.get(word)
		if word_vec is not None:
			embed_mat[idx] = numpy.array(word_vec)
			count += 1

	print "obtained pretrained values for " + str(count) + \
	      " words out of " + str(len(vocab_words)) + " words"
	return embed_mat


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Embedding matrix preparation from Glove')
	parser.add_argument('-id2w', '--id2word', type=str, default='data/id2word.json',
			    help='word dictionary')

	parser.add_argument('-glove', '--glovefile', type=str,
			    default='embedding/glove.840B.300d.txt',
			    help='whether the output tokenized version will be lowercased')

	parser.add_argument('-out', '--output', type=str, default='data/embed_mat.npy',
			    help='output file name')

	args = parser.parse_args()
	embed_mat = get_embed_mat(args.glovefile, args.id2word)
	numpy.save(args.output, embed_mat)
