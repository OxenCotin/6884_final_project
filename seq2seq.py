import random
from utils import *

import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pdb

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

# use full data for determining input/output language, in case splits somehow change it
full_data = "scan/SCAN-master/tasks.txt"
INPUT_LANG, OUTPUT_LANG, full_pairs = prepareData('scan_in', 'scan_out', full_data, False)
# used for train as well as test splits
# I add one for the <EOS> tag. Not sure if necessary but can't hurt.
INPUT_MAX = 10
MAX_LENGTH = max(len(pair[1].split(' ')) for pair in full_pairs) + 1
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Torch device: {}'.format(DEVICE))

"""
Helpers
"""

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs)) 

# Get Sentences
def indicesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indices = indicesFromSentence(lang, sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentencePadded(lang, sentence, device, max_length=MAX_LENGTH):
    indices = indicesFromSentence(lang, sentence)
    indices.append(EOS_token)
    indices.extend([PAD_token] * (max_length - len(indices)))
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)

def tensorsFromPairs(input_lang, output_lang, pairs, device):
    tensors = []
    for pair in pairs:
        input_tensor = tensorFromSentencePadded(input_lang, pair[0], device)
        target_tensor = tensorFromSentencePadded(output_lang, pair[1], device)
        tensors.append((input_tensor, target_tensor))
    return tensors


"""
Model Architecture

"""


class EncoderRNN(nn.Module):
    def __init__(self, device=None, input_size=None, hidden_size=None, dropout=.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, batch_size=1):
        # pdb.set_trace()
        embedded = self.dropout(self.embedding(input).view(1, batch_size, -1))
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, device, hidden_size, output_size, dropout=.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, batch_size=1):
        output = self.dropout(self.embedding(input).view(1, batch_size, -1))
        output = F.relu(output)
        pdb.set_trace()
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


class EncoderLSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, n_layers=1, dropout=.5):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, batch_size=1):
        # pdb.set_trace()
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        output = embedded
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        return output, (hidden, cell)

    def initHidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device), torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))


class DecoderLSTM(nn.Module):
    def __init__(self, device, hidden_size, output_size, n_layers=1, dropout=.5):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, batch_size=1):
        output = self.dropout(self.embedding(input).view(1, batch_size, -1))
        output = F.relu(output)
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        output = self.out(output.squeeze(0))
        return output, (hidden, cell)

    def initHidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device), torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))

class AttnDecoderRNN(nn.Module):
    def __init__(self, device, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch_size=1):
        embedded = self.embedding(input).view(1, batch_size, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)


"""
Training 
"""

def train(device, input_tensor, target_tensor, encoder, decoder, model, encoder_optimizer,
        decoder_optimizer, criterion, batch_size=1):

    gradient_clip = 5
    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, batch_size, encoder.hidden_size, device=device)

    loss = 0
    
    # pdb.set_trace()
    for ei in range(input_length):
        if model == "LSTM":
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], *encoder_hidden, batch_size)
        else:
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden, batch_size)
        encoder_outputs[ei] = encoder_output[0, 0]

    pdb.set_trace()
    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # pdb.set_trace()
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if model == "LSTM":
                decoder_output, decoder_hidden = decoder(
                    decoder_input, *decoder_hidden, batch_size)
            elif model == "GRU_A":
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, batch_size)
            else:
                pdb.set_trace()
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, batch_size)
            pdb.set_trace()
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if model == "LSTM":
                decoder_output, decoder_hidden = decoder(
                    decoder_input, *decoder_hidden)
            elif model == "GRU_A":
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, batch_size)
            pdb.set_trace()
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(device, encoder, decoder, model, pairs, n_iters, print_every=1000, plot_every=100,
        learning_rate=0.001, batch_size=1):
    print(f"Starting training: {n_iters} iterations")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    # evaluateRandomly(device, encoder, decoder, model, pairs, n = 1, batch_size=batch_size)

    for i in range(1, n_iters + 1):
        pdb.set_trace()
        # print("iter")
        if type(pairs) is torch.utils.data.DataLoader:
            input_tensor, target_tensor = next(iter(pairs))
            input_tensor = input_tensor.transpose(0, 1)
            target_tensor = target_tensor.transpose(0, 1)
        else:
            training_pair = tensorsFromPair(INPUT_LANG, OUTPUT_LANG, random.choice(pairs), device)
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
        loss = train(device, input_tensor, target_tensor, encoder,
                     decoder, model, encoder_optimizer, decoder_optimizer, criterion, batch_size=batch_size)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Duration (Remaining): %s Iters: (%d %d%%) Loss avg: %.4f' % (timeSince(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

            evaluateRandomly(device, encoder, decoder, model, pairs, n = 1, batch_size=batch_size)

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)
    return plot_losses


"""
Evaluation
"""


def evaluate(device, encoder, decoder, model, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(INPUT_LANG, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

        for ei in range(input_length):
            if model == "LSTM":
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         *encoder_hidden)
            else:
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        for di in range(MAX_LENGTH):
            if model == "GRU_A":
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            elif model == "LSTM":
                decoder_output, decoder_hidden = decoder(
                    decoder_input, *decoder_hidden)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(OUTPUT_LANG.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateTestSet(device, encoder, decoder, model, pairs, batch_size=1):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        hits = 0
        for pair in pairs:
            output_words = evaluate(device, encoder, decoder, model, pair[0])
            output_sentence = ' '.join(output_words)
            if output_words[-1] == '<EOS>':
                output_sentence = ' '.join(output_words[:-1])
                if pair[1] == output_sentence:
                    hits += 1
            else:
                assert len(output_words) == MAX_LENGTH, str.format(
                        'unexpected length: {} but max is {}',
                        len(output_words), MAX_LENGTH)

        print('Evaluation accuracy: {}/{} = {:.2f}%'.format(hits, len(pairs),
            hits/len(pairs)))

    encoder.train()
    decoder.train()

    return hits/len(pairs)


def evaluateRandomly(device, encoder, decoder, model, pairs, n=10, verbose=False, batch_size=1):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        hits = 0
        for i in range(n):
            if batch_size > 1:
                pdb.set_trace()
                pairs[0], pairs[1] = next(iter(pairs))
                for pair in pairs:
                    if verbose:
                        print('>', pair[0])
                        print('=', pair[1])
                    output_words = evaluate(device, encoder, decoder, model, pair[0])
                    output_sentence = ' '.join(output_words)
                    if verbose:
                        print('<', output_sentence)
                    if output_words[-1] == '<EOS>':
                        output_sentence = ' '.join(output_words[:-1])
                        if pair[1] == output_sentence:
                            hits += 1
                    if verbose:
                        print('')
            else:
                pair = random.choice(pairs)
                if verbose:
                    print('>', pair[0])
                    print('=', pair[1])
                output_words = evaluate(device, encoder, decoder, model, pair[0])
                output_sentence = ' '.join(output_words)
                if verbose:
                    print('<', output_sentence)
                if output_words[-1] == '<EOS>':
                    output_sentence = ' '.join(output_words[:-1])
                    if pair[1] == output_sentence:
                        hits += 1
                if verbose:
                    print('')

    encoder.train()
    decoder.train()

    print('Hits {}/{} test samples'.format(hits, n))


def saveModel(encoder, decoder, checkpoint, path):
    checkpoint['encoder_state_dict'] = encoder.state_dict()
    checkpoint['decoder_state_dict'] = decoder.state_dict()
    checkpoint['hidden_size'] = encoder.hidden_size
    checkpoint['dropout'] = encoder.dropout.p
    try:
        checkpoint['n_layers'] = encoder.n_layers
    except AttributeError:
        pass

    torch.save(checkpoint, path)
    print('Saved model at {}'.format(path))


def loadParameters(encoder, decoder, path):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint


def scanData(path):
    _, _, pairs = prepareData('scan_in', 'scan_out', path, False)
    print('Loaded data from {}'.format(path))
    print('{} examples. Sample pair: {}'.format(len(pairs), random.choice(pairs)))
    return pairs

def scanDataBatched(path, batch_size):
    input_lang, out_lang, pairs = prepareData('scan_in', 'scan_out', path, False)
    pairs = tensorsFromPairs(input_lang, out_lang, pairs, DEVICE)
    rand_sampler = torch.utils.data.RandomSampler(pairs, num_samples=64, replacement=True)
    batched = torch.utils.data.DataLoader(pairs, 
        batch_size=batch_size, sampler=rand_sampler)
    print('Loaded data from {}'.format(path))
    print('{} examples. Sample pair: {}'.format(len(pairs), random.choice(pairs)))
    return batched


def trainTestSplit(device, encoder, decoder, model, train_path, test_path, iters=100000, count_every=1000, batch_size=BATCH_SIZE):
    train_path = 'scan/SCAN-master/' + train_path
    test_path = 'scan/SCAN-master/' + test_path
    train_pairs = scanData(train_path)
    test_pairs = scanData(test_path)

    batched_train = scanDataBatched(train_path, batch_size)
    batched_test = scanDataBatched(test_path, batch_size)

    train_losses = trainIters(device, encoder, decoder, model, batched_train, iters, count_every, batch_size=batch_size)
    print('Evaluating training split accuracy')
    train_acc = evaluateTestSet(device, encoder, decoder, model, train_pairs)
    print('Evaluating test split accuracy')
    test_acc = evaluateTestSet(device, encoder, decoder, model, test_pairs)

    checkpoint = {'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_losses': train_losses}

    return checkpoint 


def initModel(model, device, hidden_size=200, dropout=0.5, n_layers=2):
    input_size = INPUT_LANG.n_words
    output_size = OUTPUT_LANG.n_words

    if model == 'GRU':
        encoder = EncoderRNN(device=device, input_size=input_size, hidden_size=hidden_size, dropout=dropout).to(device)
    else:
        encoder = EncoderLSTM(device, input_size, hidden_size, n_layers, dropout)

    if model == 'LSTM':
        decoder = DecoderLSTM(device, hidden_size, output_size, n_layers, dropout)
    else: # GRU
        if model == 'GRU_A':
            decoder = AttnDecoderRNN(device, hidden_size, output_size, dropout)
        else:
            decoder = DecoderRNN(device, hidden_size, output_size, dropout).to(device)

    print('Initialized model {}'.format(model))
    return encoder, decoder



def evalSplit(encoder, decoder, split_path, device):
    split_path = 'scan/SCAN-master/' + split_path
    split_pairs = scanData(split_path)
    pairs = scanData(split_path)
    accuracy = evaluateTestSet(encoder, decoder, pairs, device)
    return accuracy


if __name__ == '__main__':
    train_path = 'simple_split/tasks_train_simple.txt'
    test_path = 'simple_split/tasks_test_simple.txt'

    # model = 'GRU'
    # encoder, decoder= initModel('GRU', DEVICE, hidden_size=50, dropout=0.5)

    # model = 'GRU_A'
    # encoder, decoder = initModel('GRU_A', DEVICE, hidden_size=50, dropout=0.5)

    model = 'GRU'
    encoder, decoder = initModel(model, DEVICE, hidden_size=200, dropout=0.5, n_layers=2)

    # checkpoint_path = ''
    # loadParameters(encoder, decoder, checkpoint_path)

    checkpoint = trainTestSplit(DEVICE, encoder, decoder, model, train_path, test_path, iters=10000, count_every=1000)
    # save_path = 'simple_split_gru.pt'
    # saveModel(encoder, decoder, checkpoint, save_path) 

    # encoder, decoder, checkpoint = loadModel('simple_split1.pt', DEVICE)
    # print('Evaluating with loaded model')
    # evalSplit(encoder, decoder, test_path, DEVICE)

