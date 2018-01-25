# data preparation.
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.dont_write_bytecode = True

from imports import *
from utils import load_json, write_json, mkdir_if_not_exist
from copy import deepcopy
import argparse
import nltk
from nltk import word_tokenize as nltk_word_tokenize, sent_tokenize

from procdata import DataLoader, process_paragraphs


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='data processing pipeline for squad dataset.')
	parser.add_argument('-in', '--input', type=str, default='data/train-v1.1.json',
			    help='input file name')

	parser.add_argument('-lowercase', '--lowering', action='store_true',
			    help='whether the output tokenized version will be lowercased')

	parser.add_argument('-out', '--output', type=str, default='data/tokenized-train-v1.1.json',
			    help='output file name')

	args = parser.parse_args()

	loader = DataLoader(args.input)
	data = loader.paragraphs
	mkdir_if_not_exist(path.dirname(args.output))

	print('number of paragraphs', loader.num_paragraphs)
	print('number of QAs', loader.num_question_answers)

	if 'train' in args.input:
		datat = 'train'
	else:
		datat = 'dev'

	data = process_paragraphs(data, datatype=datat)

	write_json(data, args.output)
