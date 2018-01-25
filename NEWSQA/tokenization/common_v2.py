# data preparation.
from __future__ import print_function
from __future__ import absolute_import
from imports import *
from utils import load_json, write_json, mkdir_if_not_exist
from copy import deepcopy
import argparse
import nltk
from nltk import word_tokenize as nltk_word_tokenize, sent_tokenize


WHITESPACE_AND_PUNCTUATION = set([' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'])
ARTICLES = set(['the', 'a', 'an'])

def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        # flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def word_tokenize(sent):
    tokens = nltk.word_tokenize(sent)
    new_tokens = []
    for token in tokens:
        if token != '.' and token[-1] in ['.']:
            new_tokens.extend([token[:-1], '.'])
        elif '-' in token:
            new_tokens.extend(filter(lambda x: x, token.split('-')))
        elif '\'' in token:
            new_tokens.extend(filter(lambda x: x, token.split('\'')))
        else:
            new_tokens.append(token)
    return new_tokens


def clean_text(text, lowering):
    if isinstance(text, unicode):
        text = text.replace(u'\u00a0', ' ')
    else:
        text = text.replace('\xc2\xa0', ' ')
    text = text.replace(u'\u2019', '\'')
    text = text.replace(u'\u2013', ' - ')
    text = text.replace(u'\u2014', ' - ')
    text = text.replace("''", '" ')
    text = text.replace("``", '" ')

    while text and text[0] == ' ':
        text = text[1:]
    if text == '':
        return text

    is_unicode = lambda x: x.encode('ascii', 'ignore') == ''
    words = text.split(' ')
    num_space = 0
    for i in range(len(words) - 1, -1, -1):
        if is_unicode(words[i]):
            num_space = 1
        else:
            break
    words = [word for word in words if not is_unicode(word)]
    text = u' '.join(words)
    #    text = text.encode('ascii', 'ignore') # TODO: warning utf-8 ignore
    text += ' ' * num_space
    if lowering:
        return text.lower()
    else:
        return text


def clean_text_pre(text, lowering):
    if text == "":
        shift = 0
        return text, shift
    init_len = len(text)
    if isinstance(text, unicode):
        text = text.replace(u'\u00a0', ' ')
    else:
        text = text.replace('\xc2\xa0', ' ')
    text = text.replace(u'\u2019', '\'')
    text = text.replace(u'\u2013', ' - ')
    text = text.replace(u'\u2014', ' - ')
    text = text.replace("''", '" ')
    text = text.replace("``", '" ')

    while text and text[0] == ' ':
        text = text[1:]

    is_unicode = lambda x: x.encode('ascii', 'ignore') == ''
    words = text.split(' ')
    num_space = 0
    for i in range(len(words) - 1, -1, -1):
        if is_unicode(words[i]):
            num_space = 1
        else:
            break
    words = [word for word in words if not is_unicode(word)]
    text = u' '.join(words)
    #    text = text.encode('ascii', 'ignore') # TODO: warning utf-8 ignore
    text += ' ' * num_space
    shift = len(text) - init_len

    if lowering:
        return text.lower(), shift
    else:
        return text, shift


def clean_answer(answer, lowering):
    shift_start = 0
    answer = clean_text(answer, lowering)
    while len(answer) > 1 and answer[0] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[1:]
        shift_start += 1
    while len(answer) > 1 and answer[-1] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[:-1]
    answer = answer.split()
    if len(answer) > 1 and answer[0] in ARTICLES:
        shift_start += len(answer[0]) + 1
        answer = answer[1:]
    answer = ' '.join(answer)
    return answer, shift_start
