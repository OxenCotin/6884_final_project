from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import random

import torch

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1  

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, data, reverse=False):
    print("Reading lines...")

    pairs = []
    # Read the file and split into lines
    with open(data, mode='r') as f:
        for l in f.readlines():
            pairs.append(re.split('IN: | OUT: ', l.strip())[1:])

    # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split("")] for l in lines]
    # pairs = [re.split('IN: | OUT: ', l.strip())[1:] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
    

def prepareData(lang1, lang2, data, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, data, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



# Get Sentences
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


class ScanFn():
    def __init__(self, name, *args):
        super().__init__()
        self.name = name
        self.args = args


def parse_intermediate_rep(s): 
    def left(s, word):
        if s.index(word) == 0:
            return ''
        # get rid of space left of splitting word
        return s[0:s.index(word) - 1]

    def right(s, word):
        # get rid of splitting word, and space right of it
        return s[s.index(word) + len(word) + 1:]

    def parse(s):
        # s is the output string
        if 'after' in s:
            return ScanFn('after', 
                    parse(left(s, 'after')),
                    parse(right(s, 'after')))
        elif 'and' in s:
            return ScanFn('and', 
                    parse(left(s, 'and')),
                    parse(right(s, 'and')))
        elif 'thrice' in s:
            return ScanFn('thrice',
                    parse(left(s, 'thrice')))
        elif 'twice' in s:
            return ScanFn('twice',
                    parse(left(s, 'twice')))
        elif 'opposite' in s:
            action = parse(left(s, 'opposite'))
            turn = parse(right(s, 'opposite'))
            return ScanFn('opposite', action, turn)
        elif 'around' in s:
            action = parse(left(s, 'around'))
            turn = parse(right(s, 'around'))
            return ScanFn('around', action, turn)
        elif ' ' not in s:
            return ScanFn(s)
        elif 'left' in s:
            action = parse(left(s, 'left'))
            return ScanFn('left', action)
        elif 'right' in s:
            action = parse(left(s, 'right'))
            return ScanFn('right', action)
        else:
            assert False

    return parse(s)


def eval_rep(rep):
    def concat(*l):
        l = [a for a in l if len(a) > 0]
        return ' '.join(l)

    def get_word(s):
        if s == 'left':
            return 'LTURN'
        elif s == 'right':
            return 'RTURN'
        elif s == 'turn':
            return ''
        else:
            return s.upper()

    if rep.name == 'after':
        return concat(eval_rep(rep.args[1]), eval_rep(rep.args[0]))
    elif rep.name == 'and':
        return concat(eval_rep(rep.args[0]), eval_rep(rep.args[1]))
    elif rep.name == 'thrice':
        a = eval_rep(rep.args[0])
        return concat(a, a, a)
    elif rep.name == 'twice':
        a = eval_rep(rep.args[0])
        return concat(a, a)
    elif rep.name == 'opposite':
        action = eval_rep(rep.args[0])
        turn = eval_rep(rep.args[1])
        return concat(turn, turn, action)
    elif rep.name == 'around':
        action = eval_rep(rep.args[0])
        turn = eval_rep(rep.args[1])
        sub = concat(turn, action)
        return concat(sub, sub, sub, sub)
    elif len(rep.args) == 0:
        return get_word(rep.name)
    elif rep.name == 'left':
        action = eval_rep(rep.args[0])
        return concat('LTURN', action)
    elif rep.name == 'right':
        action = eval_rep(rep.args[0])
        return concat('RTURN', action)
    else:
        assert False


def to_string(inter_rep):
    if len(inter_rep.args) == 0: 
        return inter_rep.name
    else:
        return inter_rep.name + '(' + ', '.join(to_string(arg) for arg in
            inter_rep.args) + ')'


def polish_form(inter_rep):
    if len(inter_rep.args) == 0:
        return inter_rep.name
    else:
        return inter_rep.name + ' ' + ' '.join(polish_form(a) for a in inter_rep.args)


def parse_polish(s):

    def next(s):
        if ' ' not in s:
            return ''
        else:
            return s[s.index(' ')+1:]

    def first_word(s):
        if ' ' not in s:
            return s
        else:
            return s[0:s.index(' ')]

    # if direction follow true, we treat left/right as functions, not constants
    def parse(s, direction_follow=True):
        if s.startswith('after'):
            a1, rest = parse(next(s))
            a2, rest = parse(rest)
            return ScanFn('after', a1, a2), rest
        elif s.startswith('and'):
            a1, rest = parse(next(s))
            a2, rest = parse(rest)
            return ScanFn('and', a1, a2), rest
        elif s.startswith('thrice'):
            a1, rest = parse(next(s))
            return ScanFn('thrice', a1), rest
        elif s.startswith('twice'):
            a1, rest = parse(next(s))
            return ScanFn('twice', a1), rest
        elif s.startswith('opposite'):
            a1, rest = parse(next(s), False)
            a2, rest = parse(rest, False)
            return ScanFn('opposite', a1, a2), rest
        elif s.startswith('around'):
            a1, rest = parse(next(s), False)
            a2, rest = parse(rest, False)
            return ScanFn('around', a1, a2), rest
        elif (s.startswith('run') or s.startswith('jump') 
                or s.startswith('walk') or s.startswith('look')
                or s.startswith('turn')):
            start = first_word(s)
            return ScanFn(start), next(s)
        elif s.startswith('left') or s.startswith('right'):
            if ' ' not in s:
                return ScanFn(s), ''
            s2 = next(s)
            start = first_word(s)
            s3 = first_word(s2)
            if direction_follow and s3 in {'run', 'walk', 'jump', 'look', 'turn'}:
                a1, rest = parse(s2)
                return ScanFn(start, a1), rest
            else:
                return ScanFn(start), next(s)
        else:
            assert False, 'Hm: {}'.format(s)

    result, rest = parse(s)
    assert rest == '', 'Extra {}'.format(rest)
    return result
        
        
def test_intermediate_reps():
    # i = ' '
    # while i != '':
        # i = input()
        # print(to_string(parse_intermediate_rep(i)))
        # print(eval_rep(parse_intermediate_rep(i)))

    tasks = import_data()
    polishes = list(set([polish_form(parse_intermediate_rep(i)) for (i, o) in tasks]))
    assert len(polishes) == len(tasks)

    for (i, o) in tasks:
        inter_rep = parse_intermediate_rep(i)
        string = to_string(inter_rep)
        polish = polish_form(inter_rep)
        parsed_polish = parse_polish(polish)
        output = eval_rep(inter_rep)
        if to_string(parsed_polish) != string:
            assert False

        if output != o:
            assert False

def make_split(phrase, path):
    tasks = import_data('scan/SCAN-master/tasks.txt')

    def in_train(task):
        return phrase in task[0] and phrase != task[0]

    tasks = import_tasks(path)
    train_tasks = [task for task in tasks if in_train(task)]
    test_tasks = [task for task in tasks if not in_train(tasks)]

    export_tasks(train_tasks, path + '_train.txt')
    export_tasks(test_tasks, path + '_test.txt')


def generate_intermediate_tasks(path, new_path):
    tasks = import_data(path)

    def format_line(i, p, o):
        return 'IN: {} OUT: {}'.format(i, p, o)

    with open(new_path, 'w+') as f:
        for (i, o) in tasks:
            inter_rep = parse_intermediate_rep(i)
            string = to_string(inter_rep)
            polish = polish_form(inter_rep)
            f.write(format_line(i, string, polish) + '\n')


def import_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        # remove \n
        lines = [l[:-1] for l in lines]
        # format: IN: jump  OUT: JUMP
        tasks = []
        for line in lines:
            # put into "normal" format
            line = line.replace('I_', '')
            line = line.replace('TURN_LEFT', 'LTURN')
            line = line.replace('TURN_RIGHT', 'RTURN')

            a = line.index('IN: ') + 4
            b = line.index('OUT: ') - 1
            b2 = b + 6
            i = line[a:b]
            o = line[b2:]
            tasks.append((i, o))

        tasks = sorted(tasks, key=lambda t: len(t[1]))

    return tasks


if __name__ == '__main__':
    make_split('jump', 'scan/SCAN-master/jump')
