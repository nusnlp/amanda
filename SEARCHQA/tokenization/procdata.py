# data reader for searchqa.
from __future__ import print_function
from imports import *
from utils import load_json, write_json, mkdir_if_not_exist
from copy import deepcopy
import argparse
from common_v2 import clean_answer, clean_text, clean_text_pre, word_tokenize
from nltk import word_tokenize as nltk_word_tokenize, sent_tokenize
from tqdm import tqdm

class DataLoader(object):
	''' load data from json file '''

	def __init__(self, path):
		parsed_file = load_json(path)
		self.data = parsed_file['data']

	# self.version = parsed_file['version']

	@property
	def paragraphs(self):
		output = []
		for article in self.data:
			output.extend(article['paragraphs'])
		return output

	@property
	def num_paragraphs(self):
		return len(self.paragraphs)

	@property
	def num_question_answers(self):
		return sum([len(paragraph['qas']) for paragraph in self.paragraphs])


def clean_paragraphs(data, lowering):
	''' take in dirty paragraphs and clean them up.
	    '''
	data = deepcopy(data)
	for paragraph in data:
		context = paragraph['context']
		paragraph['context'] = clean_text(paragraph['context'], lowering)
		for qa in paragraph['qas']:
			qa['question'] = clean_text(qa['question'], lowering)
			for answer in qa['answers']:
				answer_start = answer['answer_start']
				answer_text, shift_start = clean_answer(answer['text'], lowering)
				# cleaned_text_pre,shift_clean = clean_text_pre(context[:answer_start], lowering)
				# answer['answer_start'] = len(cleaned_text_pre) + shift_clean + shift_start  # compute new start.
				answer['answer_start'] = len(clean_text(context[:answer_start], lowering)) + shift_start
				answer['text'] = answer_text
				try:
					assert (answer_text == paragraph['context'][
							       answer['answer_start']:answer['answer_start'] + len(
								       answer_text)])
				except AssertionError as e:
					print(answer_text.encode('utf-8'))
					print(paragraph['context'][
					      answer['answer_start']:answer['answer_start'] + len(answer_text)].encode(
						'utf-8'))
					print(clean_text(context, lowering).encode('utf-8') + '\n')
					print(clean_text(context[:answer_start], lowering).encode('utf-8'))

	return data


def tokenize_paragraph(paragraph):
	raw_sentences = sent_tokenize(paragraph)
	sentences = []
	for raw_sentence in raw_sentences:
		sentences.append(word_tokenize(raw_sentence))
	return sentences


def tokenize_paragraphs(data):
	''' After paragraphs have been cleaned up.
	    '''
	data = deepcopy(data)
	error_count = 0
	for paragraph in tqdm(data):
		raw_context = paragraph['context']
		context = tokenize_paragraph(raw_context)
		paragraph['context.sents'] = context
		# context = word_tokenize(raw_context)
		context = sum(context, [])
		paragraph['context.tokens'] = context
		for qa in paragraph['qas']:
			qa['question.tokens'] = word_tokenize(qa['question'])
			for raw_answer in qa['answers']:
				answer_start = raw_answer['answer_start']
				# context_pre = word_tokenize(raw_context[:answer_start])
				context_pre = sum(tokenize_paragraph(raw_context[:answer_start]), [])
				answer = word_tokenize(raw_answer['text'])
				ind = len(context_pre)
				raw_answer['text.tokens'] = answer
				raw_answer['answer_start'] = ind
				try:
					assert (answer == context[ind:ind + len(answer)])
				except AssertionError as e:
					print('\n[error %d] token mismatch answer' % error_count)
					print('context', context[ind:ind + len(answer)])
					print('answer', answer)
					error_count += 1
	return data


def clean_para(text):
	if isinstance(text, str):
		text = text.replace(u'\u00a0', ' ')
	else:
		text = text.replace('\xc2\xa0', ' ')
	text = text.replace(u'\u2019', '\'')
	text = text.replace(u'\u2013', ' - ')
	text = text.replace(u'\u2014', ' - ')
	text = text.replace("''", '" ')
	text = text.replace("``", '" ')
	return text


def process_paragraphs(data, datatype='train'):
	data = deepcopy(data)
	count = 0
	error_count = 0
	for p in data:
		# context = p["context"].split(' ')
		# context_char = list(p["context"])
		if datatype != 'train':
			p['context.tokens'] = word_tokenize(clean_para(p["context"]))
			p['context.sents'] = [word_tokenize(s) for s in sent_tokenize(clean_para(p["context"]))]
			assert len(sum(p['context.sents'], [])) == len(p['context.tokens'])
		context_pos = {}
		for qa in p["qas"]:
			question = word_tokenize(qa["question"])
			qa['question.tokens'] = question
			for a in qa['answers']:
				answer = a['text'].strip()
				answer_start = int(a['answer_start'])
				# answer_words = nltk_word_tokenize(answer+'.')
				# if answer_words[-1] == '.':
				#    answer_words = answer_words[:-1]
				# else:
				#    answer_words = nltk_word_tokenize(answer)
				answer_words = word_tokenize(clean_para(answer))
				a['text.tokens'] = answer_words

				prev_context_words = word_tokenize(clean_para(p["context"][0:answer_start]))
				left_context_words = word_tokenize(clean_para(p["context"][answer_start:]))
				answer_reproduce = []
				for i in range(len(answer_words)):
					if i < len(left_context_words):
						w = left_context_words[i]
						answer_reproduce.append(w)
				join_a = ' '.join(answer_words)
				join_ar = ' '.join(answer_reproduce)
				if join_a != join_ar:
					count += 1

				if datatype == 'train':
					p['context.tokens'] = prev_context_words + left_context_words

				a['answer_start'] = len(prev_context_words)

				try:
					assert (answer_words == p['context.tokens'][
								a['answer_start']:a['answer_start'] + len(
									answer_words)])
				except AssertionError as e:
					#print('\n[error %d] token mismatch answer' % error_count)
					#print('context', p['context.tokens'][
					#		 a['answer_start']:a['answer_start'] + len(answer_words)])
					#print('answer', answer_words)
					#print('count ', count)
					#print('---------------------------------------')
					error_count += 1

	return data

