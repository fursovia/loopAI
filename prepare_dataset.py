"""tokenize and vectorize"""

import pickle
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk import ngrams
from sklearn.model_selection import train_test_split
from model.utils import convert_to_records
from model.utils import clean
from model.utils import vectorize_text


parser = argparse.ArgumentParser()
parser.add_argument('--features', default='N', help="Whether to do some feature engineering")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--vocab_size', type=int, default=16000)
parser.add_argument('--only_words', default='Y')


if __name__ == '__main__':

    args = parser.parse_args()

    if args.only_words == 'Y':
        only_words = True
        num_words = args.vocab_size
        num_bigrams = 0
    else:
        only_words = False
        num_words = int(args.vocab_size / 2)
        num_bigrams = int(args.vocab_size / 2)

    train_raw_data_path1 = os.path.join(args.data_dir, 'initial/train_both_original.txt')
    valid_raw_data_path1 = os.path.join(args.data_dir, 'initial/valid_both_original.txt')

    train_raw_data_path2 = os.path.join(args.data_dir, 'initial/train_both_revised.txt')
    valid_raw_data_path2 = os.path.join(args.data_dir, 'initial/valid_both_revised.txt')

    with open(train_raw_data_path1, 'r', encoding='utf-8') as file:
        train_raw_data1 = file.readlines()

    with open(valid_raw_data_path1, 'r', encoding='utf-8') as file:
        valid_raw_data1 = file.readlines()

    with open(train_raw_data_path2, 'r', encoding='utf-8') as file:
        train_raw_data2 = file.readlines()

    with open(valid_raw_data_path2, 'r', encoding='utf-8') as file:
        valid_raw_data2 = file.readlines()

    train_raw_data = train_raw_data1 + train_raw_data2
    valid_raw_data = valid_raw_data1 + valid_raw_data2

    raw_data = train_raw_data + valid_raw_data

    # находим все диалоги
    dialogs = []
    personal_infos = []
    candidates = []
    all_utterances = []

    for i, line in tqdm(enumerate(raw_data)):
        if line.startswith('1 '):
            if i != 0:
                dialogs.append(dialog)
                infos = [first_persona, second_persona]
                personal_infos.append(infos)
                candidates.append(candidates_)

            first_persona = []
            second_persona = []
            dialog = []
            candidates_ = []

        if 'your persona:' in line:
            attr = ' '.join(line.split(':')[1:])
            first_persona.append(attr)
        elif 'partner\'s persona:' in line:
            attr = ' '.join(line.split(':')[1:])
            second_persona.append(attr)
        else:
            splitted_line = line[2:].split('\t')
            quest_reply = splitted_line[:2]
            cands = splitted_line[3].split('|')

            dialog.extend(quest_reply)
            all_utterances.extend(quest_reply)
            candidates_.append(cands)

    print('num of dialogs = {}'.format(len(dialogs)))
    print('num of personal infos = {}'.format(len(personal_infos)))
    print('num of candidates = {}'.format(len(candidates)))

    print('cleaning utterances...')
    all_utterances_cleaned = []
    for utt in tqdm(all_utterances):
        all_utterances_cleaned.append(clean(utt))

    # чистим диалоги и персональную инфу
    cleaned_dials = []
    cleaned_infos = []
    cleaned_cands = []

    for i, dialog in enumerate(tqdm(dialogs)):
        cleaned_dial = []
        first_cleaned = []
        second_cleaned = []
        first_info, second_info = personal_infos[i]

        for info in first_info:
            cleaned_info = clean(info)
            first_cleaned.append(cleaned_info)

        for info in second_info:
            cleaned_info = clean(info)
            second_cleaned.append(cleaned_info)

        cleaned_infos.append([first_cleaned, second_cleaned])

        for idx, mes in enumerate(dialog):
            cleaned_mes = clean(mes)
            cleaned_dial.append(cleaned_mes)
        cleaned_dials.append(cleaned_dial)

        curr_cands = candidates[i]
        cleaned_cands_ = []
        for cand in curr_cands:
            cleaned_mes = []
            for mes in cand:
                cleaned_mes.append(clean(mes))
            cleaned_cands_.append(cleaned_mes)
        cleaned_cands.append(cleaned_cands_)

    print('num of dialogs = {}'.format(len(cleaned_dials)))
    print('num of personal infos = {}'.format(len(cleaned_infos)))
    print('num of candidates = {}'.format(len(cleaned_cands)))

    pickle.dump(cleaned_dials, open(os.path.join(args.data_dir, 'dials.pkl'), 'wb'))
    pickle.dump(cleaned_infos, open(os.path.join(args.data_dir, 'infos.pkl'), 'wb'))
    pickle.dump(cleaned_cands, open(os.path.join(args.data_dir, 'cands.pkl'), 'wb'))

    # диалог начинает другой человек, а мы отвечаем
    X = []
    Y = []

    for i, dialog in enumerate(tqdm(cleaned_dials)):
        for idx, mes in enumerate(dialog):
            if idx % 2 == 0:
                personal_info = cleaned_infos[i][1]
            else:
                personal_info = cleaned_infos[i][0]

            if idx == 1:
                context = ''
            if idx == 2:
                context_len = 1
                context = ' '.join(dialog[(idx - 1 - context_len):(idx - 1)])
            if idx == 3:
                context_len = 2
                context = ' '.join(dialog[(idx - 1 - context_len):(idx - 1)])
            if idx > 3:
                context_len = 3
                context = ' '.join(dialog[(idx - 1 - context_len):(idx - 1)])

            if idx >= 1:
                question = dialog[idx - 1]
                reply = dialog[idx]
                x_small = [context, question, reply, personal_info]
                X.append(x_small)
                Y.append(1)

                if idx % 2 == 1:
                    idxx = int(idx/2)
                    neg_cands = cleaned_cands[i]
                    curr_negs = neg_cands[idxx]  # 20 кандидатов
                    for cand in curr_negs:
                        x_small = [context, question, cand, personal_info]
                        X.append(x_small)
                        Y.append(0)
                # VERY SLOW and no need for this since we already have unbalanced classes
                # else:
                #     curr_negs = np.random.choice(all_utterances_cleaned, 20, replace=False)
                #     for cand in curr_negs:
                #         x_small = [context, question, cand, personal_info]
                #         X.append(x_small)
                #         Y.append(0)

    # TODO: придумать фичи
    if args.features == 'Y':
        pass

    corpus = []
    for i in range(len(cleaned_dials)):
        dial = cleaned_dials[i]
        infos = cleaned_infos[i]
        cands_ = cleaned_cands[i]
        for mes in dial:
            corpus.append(mes)
        for inf_ in infos:
            for mes in inf_:
                corpus.append(mes)
        for c in cands_:
            for mes in c:
                corpus.append(mes)

    corpus = list(set(corpus))

    # получаем словарь
    word2idx_path = os.path.join(args.data_dir, 'word2idx.pkl')
    if os.path.isfile(word2idx_path):
        print('Loading word2idx file from {}'.format(word2idx_path))
        word2idx = pickle.load(open(word2idx_path, 'rb'))
    else:
        words = ' '.join(corpus).split()
        bigrams = ngrams(words, 2)

        uni_counter = Counter(words)
        bi_counter = Counter(bigrams)

        print('Num of unigrams: {0}'.format(len(uni_counter.most_common())))
        print('Num of bigrams: {0}'.format(len(bi_counter.most_common())))

        uni2idx = {word[0]: i + 1 for i, word in enumerate(uni_counter.most_common(num_words))}
        if not only_words:
            bi2idx = {word[0]: num_words + 1 + i for i, word in enumerate(bi_counter.most_common(num_bigrams))}
        else:
            bi2idx = {}

        word2idx = {}
        word2idx.update(uni2idx)
        word2idx.update(bi2idx)

        pickle.dump(word2idx, open(word2idx_path, 'wb'))
        pickle.dump(corpus, open(os.path.join(args.data_dir, 'corpus.pkl'), 'wb'))
        print('word2idx file saved at {}'.format(word2idx_path))
        print('corpus file saved at {}'.format(args.data_dir))

    print('Vectorizing...')
    # векторизуем текст
    context_vect = []
    question_vect = []
    reply_vect = []
    info_vect = []

    for i, dial in enumerate(tqdm(X)):
        cont = vectorize_text(dial[0], word2idx, 60, 'pre')
        ques = vectorize_text(dial[1], word2idx)
        reply = vectorize_text(dial[2], word2idx)

        context_vect.append(cont)
        question_vect.append(ques)
        reply_vect.append(reply)

        info_ = dial[3]
        for j in range(5):  # 5 фактов о каждом
            try:
                vect_info = vectorize_text(info_[j], word2idx)
                info_vect.append(vect_info)
            except IndexError:
                info_vect.append(np.zeros_like(ques))  # длина вопроса равна длине факта

    data = np.hstack((context_vect, question_vect, reply_vect, np.array(info_vect).reshape(-1, 100)))
    Y = np.array(Y, int)

    Ytr, Yev, Xtr, Xev = train_test_split(Y,
                                          data,
                                          test_size=0.1,
                                          random_state=24)

    sample_path = os.path.join(args.data_dir, 'sample')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    print('Converting to TFRecords...')
    convert_to_records([data, Y], 'full', args.data_dir)
    convert_to_records([Xtr, Ytr], 'train', args.data_dir)
    convert_to_records([Xev, Yev], 'eval', args.data_dir)

    convert_to_records([data[:1000], Y[:1000]], 'full', sample_path)
    convert_to_records([Xtr[:1000], Ytr[:1000]], 'train', sample_path)
    convert_to_records([Xev[:1000], Yev[:1000]], 'eval', sample_path)
