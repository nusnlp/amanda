import re
import json
import argparse
from tqdm import tqdm
import os


def dump_dict(fname, data):
	with open(fname, 'w') as fp:
		json.dump(data, fp)


def get_lines(fname):
	with open(fname, 'r') as fp:
		lines = fp.readlines()
	return lines


def clean_lines(lines):
	return [line.strip() for line in lines]


def cut_context(whole_text, answer, maxlen):
	def get_ctx_words(whole_text, answer, start_ind):
		prev_words = whole_text[:start_ind].split()
		next_words = whole_text[start_ind + len(answer):].split()
		ans_words = answer.split()
		if len(prev_words) >= maxlen / 2.0 and len(next_words) >= maxlen / 2.0:
			ctx_words = prev_words[- int(maxlen / 2):] + \
						ans_words + \
						next_words[:int(maxlen / 2)]
		elif len(prev_words) >= maxlen / 2.0 and len(next_words) < maxlen / 2.0:
			ctx_words = prev_words[
				    -min(len(prev_words), maxlen - len(next_words)):] + \
						ans_words + next_words
		elif len(prev_words) < maxlen / 2.0 and len(next_words) >= maxlen / 2.0:
			ctx_words = prev_words + ans_words + \
						next_words[:min(len(next_words), maxlen - len(prev_words))]
		else:
			ctx_words = prev_words + ans_words + next_words
		return ctx_words

	if answer not in whole_text and answer.lower() not in whole_text.lower():
		whole_text_words = whole_text.split()
		return ' '.join(whole_text_words[:min(maxlen, len(whole_text_words))])
	else:
		if ' ' + answer + ' ' in whole_text:
			# whole_text_words = whole_text.split()
			start_ind = whole_text.find(' ' + answer + ' ')
			ctx_words = get_ctx_words(whole_text, answer, start_ind+1)
		elif answer + ' ' in whole_text:
			start_ind = whole_text.find(answer + ' ')
			ctx_words = get_ctx_words(whole_text, answer, start_ind)
		elif answer in whole_text:
			start_ind = whole_text.find(answer)
			ctx_words = get_ctx_words(whole_text, answer, start_ind)
		elif answer.lower() in whole_text.lower():
			start_ind = whole_text.lower().find(answer.lower)
			ctx_words = get_ctx_words(whole_text, answer, start_ind)
		else:
			whole_text_words = whole_text.split()
			ctx_words = whole_text_words[:min(maxlen, len(whole_text_words))]
		return ' '.join(ctx_words)


def get_ans_start(text, ans):
	if ans in text:
		return text.find(ans)
	elif ans.lower() in text.lower():
		return text.lower().find(ans.lower())
	else:
		return 0


def prepare_data_dict_for_training(txtf, maxlen=150):
	datadict = {}
	datadict['version'] = 'v1'
	datadict['data'] = []
	datatype = txtf[:-4]
	lines = get_lines(txtf)
	lines = clean_lines(lines)
	count = 0
	for line in tqdm(lines):
		article = {}
		article['title'] = 'NONE'
		paragraph = {}
		qid = 'searchqa_' + datatype + '_' + str(count)
		ctx, q, ans = line.split('|||')
		ctx = ctx.strip()
		q = q.strip()
		ans = ans.strip()
		paras = re.findall(r'<s>(.*?)<\/s>', ctx, re.I|re.S)
		paras = clean_lines(paras)
		whole_text = ' '.join(paras)
		# fixing the extra spaces
		whole_text = ' '.join(whole_text.split())
		short_text = cut_context(whole_text, ans, maxlen)
		paragraph['context'] = short_text
		paragraph['snippet_list'] = paras
		qa = {}
		qa['id'] = qid
		qa['question'] = q
		qa['answers'] = [{'answer_start': get_ans_start(short_text, ans) , 'text': ans}]
		paragraph['qas'] = [qa]
		article['paragraphs'] = [paragraph]
		datadict['data'].append(article)
		count += 1
	return datadict


def prepare_data_dict_for_val_test(txtf):
	datadict = {}
	datadict['version'] = 'v1'
	datadict['data'] = []
	datatype = txtf[:-4]
	lines = get_lines(txtf)
	lines = clean_lines(lines)
	count = 0
	for line in tqdm(lines):
		article = {}
		article['title'] = 'NONE'
		paragraph = {}
		qid = 'searchqa_' + datatype + '_' + str(count)
		ctx, q, ans = line.split('|||')
		ctx = ctx.strip()
		q = q.strip()
		ans = ans.strip()
		paras = re.findall(r'<s>(.*?)<\/s>', ctx, re.I|re.S)
		paras = clean_lines(paras)
		whole_text = ' '.join(paras)
		# fixing the extra spaces
		whole_text = ' '.join(whole_text.split())
		# short_text = cut_context(whole_text, ans, maxlen)
		paragraph['context'] = whole_text
		paragraph['snippet_list'] = paras
		qa = {}
		qa['id'] = qid
		qa['question'] = q
		qa['answers'] = [{'answer_start': get_ans_start(whole_text, ans) , 'text': ans}]
		paragraph['qas'] = [qa]
		article['paragraphs'] = [paragraph]
		datadict['data'].append(article)
		count += 1
	return datadict


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='convert the SearchQA data format.')
	parser.add_argument('-trainf', '--trainfile', type=str, default='data/train.txt',
			    help='text format file of SearchQA training data')
	parser.add_argument('-valf', '--valfile', type=str, default='data/val.txt',
						help='text format file of SearchQA validation data')
	parser.add_argument('-testf', '--testfile', type=str, default='data/test.txt',
						help='text format file of SearchQA test data')
	parser.add_argument('-out', '--outdir', type=str, default='data',
			    help='output directory for the json files to dump.')
	args = parser.parse_args()

	print "Preparing the data"
	tr_datadict = prepare_data_dict_for_training(args.trainfile, maxlen=400)
	val_datadict = prepare_data_dict_for_val_test(args.valfile)
	test_datadict = prepare_data_dict_for_val_test(args.testfile)

	print "Dumping the data"
	dump_dict(os.path.join(args.outdir, 'train.json'), tr_datadict)
	dump_dict(os.path.join(args.outdir, 'val.json'), val_datadict)
	dump_dict(os.path.join(args.outdir, 'test.json'), test_datadict)
