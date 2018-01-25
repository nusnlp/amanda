import json
import re
import pandas as pd
import ast
import sys

csv_data_path = sys.argv[1]
json_fname = sys.argv[2]


data = pd.read_csv(csv_data_path, encoding='utf-8')

story_ids = data['story_id'].values.tolist()
questions = data['question'].values.tolist()
answer_char_rangess = data['answer_char_ranges'].values.tolist()
is_answer_absents = data['is_answer_absent'].values.tolist()
is_question_bads = data['is_question_bad'].values.tolist()
validated_answerss = data['validated_answers'].values.tolist()
story_texts = data['story_text'].values.tolist()


s_data = dict()
s_data['version'] = 'v-1.1'

article = dict()
article['paragraphs'] = []
for sidx in range(len(story_ids)):
    sid = story_ids[sidx]
    paragraph = dict()
    paragraph['context'] = story_texts[sidx]
    #paragraph['qas'] = []
    qans = dict()
    qans['id'] = str(sidx)
    qans['question'] = questions[sidx]
    qans['validated_answers'] = validated_answerss[sidx]
    qans['answers'] = []
    answer_char_ranges = re.split('[|,]', answer_char_rangess[sidx])
    for acr in answer_char_ranges:
        ans_dict = dict()
        if acr != 'None':
            astart = acr.split(':')[0]
            aend = acr.split(':')[1]
            ans_dict['answer_start'] = astart
            ans_dict['text'] = story_texts[sidx][int(astart):int(aend)]
            qans['answers'].append(ans_dict)

    valid_qans = dict()
    if str(validated_answerss[sidx]) == 'nan':
        valid_qans = qans['answers'][0]
    else:
        v_d = ast.literal_eval(validated_answerss[sidx])
        valid_ans_range = v_d.keys()[v_d.values().index(max(v_d.values()))]
        if valid_ans_range in ['none', 'bad_question']:
            valid_qans = qans['answers'][0]
        else:
            v_start = valid_ans_range.split(':')[0]
            v_end = valid_ans_range.split(':')[1]
            valid_qans['answer_start'] = v_start
            valid_qans['text'] = story_texts[sidx][int(v_start):int(v_end)]

    qans['answer'] = [valid_qans]
    paragraph['qas'] = [qans]
    article['paragraphs'].append(paragraph)

    if (sidx+1)%1000 == 0:
        print("Done for " + str(sidx+1) + " samples..")

s_data['data'] = [article]

print("Dumping data...")
with open(json_fname, 'w') as fp:
    json.dump(s_data, fp)
