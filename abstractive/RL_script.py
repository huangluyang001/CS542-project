from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time, math
import preprocess4model
from pyrouge import Rouge155
import pickle, os
from modules.train import trainIters, validation
from modules.model import EncoderRNN, concatAttnDecoderRNN, BahdanauAttnDecoderRNN
from nltk import sent_tokenize
import copy
from rouge.rouge_scorer import RougeScorer


id2tag = {
0: 'science',
1: 'government',
2: 'arts',
3: 'law',
4: 'business',
5: 'sports',
6: 'lifestyle',
7:'education',
8: 'health',
9: 'technology',
10: 'military',
}


device = torch.device('cuda')


PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 400
MAX_TARGET = 100

class Lang:
    def __init__(self, name, word2index, index2word):
        self.name = name
        self.word2index = word2index
        self.word2count = {}
        self.index2word = index2word
        self.n_words = len(word2index)



def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def concat_seq(seq, max_length):
    return seq[:max_length]



def beam_evaluate(encoder, decoder, input_tensor, input_length, output_lang, max_length=MAX_LENGTH, max_target_length=100, device='cuda', beam_size=3):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device).view(1, -1)
        #input_length = input_tensor.size()[0]

        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)


        encoder_outputs, encoder_hidden = encoder(input_tensor,
                                                 input_length,
                                                  None)
        # encoder_outputs[ei] += encoder_output[0, 0]


        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        prob = F.log_softmax(decoder_output, dim=-1)
        topvs, topis = prob.topk(beam_size)
        topvs = topvs.view(-1)
        topis = topis.view(-1)
        candidate_list = []
        for j in range(len(topvs)):
            candidate_list.append((topvs[j].item(), [int(topis[j].item())], decoder_hidden))

        result = []


        for di in range(max_target_length-1):
            new_candidate_list = []
            for ly_i in range(len(candidate_list)):
                candidate = candidate_list[ly_i]

                if candidate[1][-1] == EOS_token:
                    result.append((candidate[0], candidate[1]))
                else:
                    new_candidate_list.append(candidate)
            candidate_list = new_candidate_list
            new_candidate_list = []
                    # word_list = []
                    # for id in candidate[1]:
                    #     word_list.append(output_lang.index2word[id])
                    # return word_list
            if len(result) > beam_size - 1:
                break

            for crt_candidate in candidate_list:
                decoder_hidden = crt_candidate[2]
                decoder_input = torch.tensor([[crt_candidate[1][-1]]], device=device)
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                prob = F.log_softmax(decoder_output, dim=-1).view(-1)
                #decoder_attentions[:,di] = decoder_attention.data
                topvs, topis = prob.topk(beam_size)

                for jjj in range(len(topis)):
                    topi = topis[jjj]
                    topv = topvs[jjj]
                    word = int(topi.item())
                    old_word_list = copy.deepcopy(crt_candidate[1])
                    old_word_list.append(word)
                    new_candidate_list.append((crt_candidate[0]+topv.item(), old_word_list, decoder_hidden))
            new_candidate_list = sorted(new_candidate_list, key=lambda x:x[0], reverse=True)
            candidate_list = new_candidate_list[:3]
        if len(result) < 3:
            for i in range(3-len(result)):
                candidate = candidate_list[i]
                result.append((candidate[0], candidate[1]))
    result = sorted(result, key=lambda x:x[0], reverse=True)


    word_list = []
    for id in result[0][1]:
        word_list.append(output_lang.index2word[id])


    encoder.train(True)
    decoder.train(True)


    return word_list

def beam_evaluate_unk(encoder, decoder, input_tensor, input_length, output_lang, max_length=MAX_LENGTH,
                      max_target_length=100, min_target_length=20, device='cuda', beam_size=5, printnum=3, block_ngram_repeat=3, pre_input=[]):
    exclusion_tokens = set([4, 5])
    new_beam_size = beam_size
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        input_tensor = torch.tensor(input_tensor, dtype=torch.long, device=device).view(1, -1)
        #input_length = input_tensor.size()[0]

        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)


        encoder_outputs, encoder_hidden = encoder(input_tensor,
                                                 input_length,
                                                  None)
        # encoder_outputs[ei] += encoder_output[0, 0]


        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        h_size = (1, 1, decoder.hidden_size)
        input_feed = encoder_hidden[0].data.new(*h_size).zero_()
        decoder_output, decoder_hidden, decoder_attention, input_feed = decoder(
            decoder_input, decoder_hidden, encoder_outputs, input_feed)
        prob = F.log_softmax(decoder_output, dim=-1)
        topvs, topis = prob.topk(beam_size)
        topvs = topvs.view(-1)
        topis = topis.view(-1)
        candidate_list = []
        for j in range(len(topvs)):
            if int(topis[j].item()) == UNK_token:
                _, encode_index = decoder_attention.topk(1, dim=0)
                if pre_input != []:
                    crt_token = pre_input[encode_index.view(-1).item()]
                else:
                    crt_token = input_tensor[0, encode_index.view(-1).item()].view(-1).item()
                    crt_token = int(crt_token)
                next_word = int(topis[j].item())
                # if crt_token == UNK_token:
                #     _, encode_index = decoder_attention.topk(2, dim=0)
                #     encode_index = encode_index.view(-1)[1].item()
                #     crt_token = input_tensor[0, encode_index].view(-1)
                candidate_list.append((topvs[j].item(), [crt_token], decoder_hidden, next_word, input_feed, set()))
            else:
                candidate_list.append((topvs[j].item(), [int(topis[j].item())], decoder_hidden, int(topis[j].item()), input_feed, set()))

        result = []


        for di in range(max_target_length+40-1):
            new_candidate_list = []

            for ly_i in range(len(candidate_list)):
                if len(new_candidate_list) > new_beam_size - 1:
                    break
                candidate = candidate_list[ly_i]
                lengthofsent = len([i for i in candidate[1] if i != 4 and i != 5])

                if candidate[1][-1] == EOS_token:
                    if lengthofsent > min_target_length:
                        result.append((candidate[0], candidate[1]))
                        new_beam_size = new_beam_size - 1
                elif lengthofsent > max_target_length - 1:
                    result.append((candidate[0], candidate[1]))
                    new_beam_size = new_beam_size - 1
                else:
                    new_candidate_list.append(candidate)

            candidate_list = copy.deepcopy(new_candidate_list)
            new_candidate_list = []
            if len(result) > beam_size - 1:
                break

            id = 0
            for crt_candidate in candidate_list:
                decoder_hidden = crt_candidate[2]
                input_feed = crt_candidate[4]
                n_grams = crt_candidate[5]
                decoder_input = torch.tensor([[crt_candidate[3]]], device=device)
                decoder_output, decoder_hidden, decoder_attention, input_feed = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, input_feed)
                prob = F.log_softmax(decoder_output, dim=-1).view(-1)
                #decoder_attentions[:,di] = decoder_attention.data
                topvs, topis = prob.topk(beam_size)

                for jjj in range(len(topis)):
                    topi = topis[jjj]
                    topv = topvs[jjj]
                    word = int(topi.item())
                    if word == UNK_token:
                        _, encode_index = decoder_attention.topk(1, dim=0)
                        if pre_input != []:
                            crt_word = pre_input[encode_index.view(-1).item()]
                        else:
                            crt_word = input_tensor[0, encode_index.view(-1).item()].view(-1)
                            crt_word = crt_word.item()
                        # if crt_word == UNK_token:
                        #     _, encode_index = decoder_attention.topk(2, dim=0)
                        #     encode_index =  encode_index.view(-1)[1].item()
                        #     crt_word = input_tensor[0, encode_index].view(-1)
                        old_word_list = copy.deepcopy(crt_candidate[1])
                        old_word_list.append(crt_word)
                        new_candidate_list.append((crt_candidate[0] + topv.item(), old_word_list, decoder_hidden, word, input_feed, n_grams))
                    else:
                        old_word_list = copy.deepcopy(crt_candidate[1])
                        old_word_list.append(word)
                        new_candidate_list.append((crt_candidate[0]+topv.item(), old_word_list, decoder_hidden, word, input_feed, n_grams))
                id += 1
                if id > new_beam_size - 1:
                    break
            if block_ngram_repeat > 0:
                temp_candidate_list = []
                for crt_candidate in new_candidate_list:
                    prob = crt_candidate[0]
                    word_list = crt_candidate[1]
                    ngrams = copy.deepcopy(crt_candidate[5])
                    gram = word_list[-block_ngram_repeat:]
                    if set(gram) & exclusion_tokens:
                        temp_candidate_list.append(
                            (prob, word_list, crt_candidate[2], crt_candidate[3], crt_candidate[4], ngrams))
                        continue
                    if tuple(gram) in ngrams:
                        prob = -10e20
                    ngrams.add(tuple(gram))
                    temp_candidate_list.append((prob, word_list, crt_candidate[2], crt_candidate[3], crt_candidate[4], ngrams))
                # print(len(new_candidate_list), len(temp_candidate_list))
                # print(new_candidate_list[0][0], new_candidate_list[0][1], new_candidate_list[0][5], tuple(new_candidate_list[0][1][-3:]) in new_candidate_list[0][5])
                # print(temp_candidate_list[0][0], temp_candidate_list[0][1], temp_candidate_list[0][5])
                new_candidate_list = copy.deepcopy(temp_candidate_list)

            new_candidate_list = sorted(new_candidate_list, key=lambda x:x[0], reverse=True)
            candidate_list = copy.deepcopy(new_candidate_list)

        if len(result) < beam_size:
            for i in range(beam_size-len(result)):
                try:
                    candidate = candidate_list[i]
                    result.append((candidate[0], candidate[1]))
                except IndexError:
                    print(candidate_list)
                    break
    result = sorted(result, key=lambda x:x[0], reverse=True)
    #result = sorted(result, key=lambda x:x[0], reverse=True)

    sent_list = []
    word_list = []
    for j in range(printnum):
        # print(j)
        # print(len(result))
        # print(len(new_candidate_list))
        # print(new_beam_size)
        # print(beam_size)
        # print(candidate_list)
        for id in result[j][1]:
            try:
                word_list.append(output_lang.index2word[id])
            except KeyError:
                word_list.append(id)
        sent_list.append(word_list)
        word_list = []


    encoder.train(True)
    decoder.train(True)


    return sent_list[0]


def beam_evaluate_unk_new(encoder, decoder, input_tensor, input_length, output_lang, max_length=MAX_LENGTH,
                      max_target_length=100, min_target_length=20, device='cuda', beam_size=5, printnum=3, block_ngram_repeat=3):
    pass

def encoding(sent_list, tk2id, max_length):
    inputs = []
    input_lengths = []
    for sent in sent_list:
        indexes = [tk2id[token] if tk2id.__contains__(token) else UNK_token for token in sent]
        indexes.append(EOS_token)
        indexes = concat_seq(indexes, max_length)
        input_lengths.append(len(indexes))
        indexes = pad_seq(indexes, max_length)
        inputs.append(indexes)
    return (inputs, input_lengths)


def quick_rouge(encoder1, attn_decoder1, test_input_list, test_output_list, in_tk2id, out_lang, max_input=100, min_target_length=0):
    test_input, test_input_lengths = encoding(test_input_list, in_tk2id, max_length=max_input)
    length = len(test_input)
    #length = 1000
    # test_filename = 'testout/test-bl.txt'
    # gold_truth = 'testout/gold.txt'
    rouge_scores = 0
    rouge_1 = 0
    rouge_2 = 0
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for i in range(length):
        # if i < 7698:
        #     continue
        sent = test_input[i]
        true = test_output_list[i]
        input_length = [test_input_lengths[i]]
        #prediction,_ = evaluate(encoder1, attn_decoder1,  sent, input_length, out_lang, max_length=MAX_LENGTH)
        prediction = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=1, min_target_length=min_target_length, printnum=1, block_ngram_repeat=0, max_target_length=max_output)
        # prediction_2 = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=5, min_target_length=20, block_ngram_repeat=0, printnum=1)
        if prediction[-1] == 'EOS':
            prediction.pop()
        else:
            pass
            #print(len(prediction))
        prediction_new = [i for i in prediction if not (i == '<t>')]
        true_new = [i for i in true if not (i == '<t>')]
        if prediction_new == []:
            prediction_new = ['UNK']
        scores = scorer.score(' '.join(true_new), ' '.join(prediction_new))
        rouge_scores += scores['rougeL'][2]
        rouge_1 += scores['rouge1'][2]
        rouge_2 += scores['rouge2'][2]
    print('Rouge-L', rouge_scores / length)
    print('Rouge-1', rouge_1 / length)
    print('Rouge-2', rouge_2 / length)



if __name__ == '__main__':
    torch.cuda.set_device(0)
    torch.manual_seed(128)
    train_label = True
    test = False
    inference = True
    preprocess = preprocess4model.Preprocessing()
    if not test:
        # filename_test = 'test_processed_topic.pkl'
        # test_filename = 'testout/test-bl.txt'
        # gold_truth = 'testout/gold-bl.txt'
        filename_test = 'test.pkl'
        test_filename = 'testout/test-rl.txt'
        gold_truth = 'testout/gold-rl.txt'
    if test:
        filename = 'train_mini.pkl'
        filename_dvp = 'vld_mini.pkl'
        train_file = open(filename, 'rb')
        train_all = pickle.load(train_file)
        in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list, _ = train_all
        train_file.close()
        # test_input_list, test_output_list = preprocess.test_preprocess_all(filename='test_nyt.pkl',
        #                                                                                numofdata=20)
        # validation_input_list, validation_output_list = preprocess.test_preprocess_all(filename='dvp_nyt.pkl',
        #                                                                                numofdata=50000)
        train_file = open(filename_dvp, 'rb')
        train_all = pickle.load(train_file)
        _, _, _, _, validation_input_list, validation_output_list, _ = train_all
        train_file.close()
        input_list = input_list[:10]
        output_list = output_list[:10]
        validation_input_list = input_list[:10]
        validation_output_list = output_list[:10]
        print("Loaded")
        print(len(in_tk2id))
        print(len(out_id2tk))
        print('Development num: ', len(validation_input_list))
        print('Train num: ', len(input_list))
        print(in_id2tk[0])
        print(in_id2tk[1])
        print(in_id2tk[2])
        print(in_id2tk[3])
        print(in_id2tk[4])
        print(in_id2tk[5])
        batch_size = 10
        init_accum = 0.
        learning_rate = 0.001
        steps = 20
        print_every = 5
    else:
        filename = 'train.pkl'
        #filename = 'train_mini.pkl'
        filename_dvp = 'val.pkl'
        train_file = open(filename, 'rb')
        train_all = pickle.load(train_file)
        in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list= train_all
        train_file.close()
        # test_input_list, test_output_list = preprocess.test_preprocess_all(filename='test_nyt.pkl',
        #                                                                                numofdata=20)
        # validation_input_list, validation_output_list = preprocess.test_preprocess_all(filename='dvp_nyt.pkl',
        #                                                                                numofdata=50000)
        train_file = open(filename_test, 'rb')
        train_all = pickle.load(train_file)
        _, _, _, _, test_input_list, test_output_list = train_all
        train_file.close()
        train_file = open(filename_dvp, 'rb')
        train_all = pickle.load(train_file)
        _, _, _, _, validation_input_list, validation_output_list= train_all
        #num = random.randint(2,100000)
        #print(num)
        # input_list, output_list = input_list[:1000], output_list[:1000]
        # input_list, output_list = [input_list[4356]], [output_list[4356]]
        # validation_input_list, validation_output_list = input_list[:1000], output_list[:1000]
        # test_input_list, test_output_list = input_list[:1000], output_list[:1000]
        train_file.close()
        print("Loaded")
        print(len(in_tk2id))
        print(len(out_id2tk))
        print('Test num: ', len(test_input_list))
        print('Development num: ', len(validation_input_list))
        print('Train num: ', len(input_list))
        print(in_id2tk[0])
        print(in_id2tk[1])
        print(in_id2tk[2])
        print(in_id2tk[3])
        print(in_id2tk[4])
        print(in_id2tk[5])
        batch_size = 10
        learning_rate = 1e-4
        init_accum = 0.1
        steps = 1000
        print_every = 50
        max_input = 400
        max_output = 100
        Initialization = True
    in_lang = Lang('nyt_in', in_tk2id, in_id2tk)
    out_lang = Lang('nyt_out', out_tk2id, out_id2tk)
    embed_size = 128
    hidden_size = 512
    enc_hidden_size = 256 # hidden_size/2 if bidirectional
    attn_size = 512
    print(len(in_tk2id))
    print(len(in_id2tk))
    assert len(in_tk2id) == len(out_tk2id)
    embedding = nn.Embedding(len(in_tk2id), embed_size, padding_idx=0)
    encoder1 = EncoderRNN(embedding, in_lang.n_words, embed_size, enc_hidden_size, device=device).to(device)
    attn_decoder1 = BahdanauAttnDecoderRNN(embedding, embed_size, hidden_size, attn_size, out_lang.n_words, dropout_p=0.,
                                         device=device).to(device)
    #encoder_optimizer = optim.Adagrad(encoder1.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    #decoder_optimizer = optim.Adagrad(attn_decoder1.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    encoder_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(attn_decoder1.parameters(), lr=learning_rate)
    print(encoder1)
    print(attn_decoder1)
    epoch = 0
    ckp_path = 'checkpoint/'
    checkpoint_name_base = 'checkpoint_50k_'
    rl_checkpoint_name_base = 'checkpoint_50k_rl_'
    if Initialization:
        if not os.path.exists(ckp_path):
            os.mkdir(ckp_path)
        try:
            checkpoint_name = ckp_path + checkpoint_name_base + '_0.tar'
            checkpoint = torch.load(checkpoint_name)
            epoch = checkpoint['epoch'] + 1
            print(epoch)
            encoder1.load_state_dict(checkpoint['encoder_state_dict'])
            attn_decoder1.load_state_dict(checkpoint['decoder_state_dict'])
            #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            print('weight loaded ml')
        except FileNotFoundError:
            print('Start New Train')
    else:
        if not os.path.exists(ckp_path):
            os.mkdir(ckp_path)
        try:
            checkpoint_name = ckp_path + rl_checkpoint_name_base + '_9.tar'
            checkpoint = torch.load(checkpoint_name)
            epoch = checkpoint['epoch'] + 1
            print(epoch)
            encoder1.load_state_dict(checkpoint['encoder_state_dict'])
            attn_decoder1.load_state_dict(checkpoint['decoder_state_dict'])
            #encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            #decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            print('weight loaded rl')
        except FileNotFoundError:
            raise Exception('Wrong rl checkpoint name')
    # Prepare Data
    input_1, input_lengths_1 = encoding(input_list, in_tk2id, max_length=50)
    input_2, input_lengths_2 = encoding(input_list, in_tk2id, max_length=200)
    summary_1, summary_lengths_1 = encoding(output_list, out_tk2id, max_length=50)
    # if True:
    #     test_input, test_input_lengths = encoding(test_input_list, in_tk2id, max_length=max_input)
    #     length = len(test_input)
    #     #length = 1000
    #     # test_filename = 'testout/test-bl.txt'
    #     # gold_truth = 'testout/gold.txt'
    #     f1 = open(test_filename, 'w', encoding='utf-8')
    #     f2 = open(gold_truth, 'w', encoding='utf-8')
    #     rouge_scores = 0
    #     scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    #     for i in range(length):
    #         sent = test_input[i]
    #         true = test_output_list[i]
    #         input_length = [test_input_lengths[i]]
    #         #prediction,_ = evaluate(encoder1, attn_decoder1,  sent, input_length, out_lang, max_length=MAX_LENGTH)
    #         prediction = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=1, min_target_length=0, printnum=1, block_ngram_repeat=0, max_target_length=max_output)
    #         # prediction_2 = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=5, min_target_length=20, block_ngram_repeat=0, printnum=1)
    #         if prediction[-1] == 'EOS':
    #             prediction.pop()
    #         else:
    #             pass
    #             #print(len(prediction))
    #         prediction_new = [i for i in prediction if not (i == '<t>')]
    #         true_new = [i for i in true if not (i == '<t>')]
    #         if prediction_new == []:
    #             prediction_new = ['UNK']
    #         scores = scorer.score(' '.join(true_new), ' '.join(prediction_new))
    #         rouge_scores += scores['rougeL'][2]
    #         if (i+1) % 1000 == 0:
    #             print('rouge temp: ', rouge_scores/(i+1))
    #     print('Rouge-L', rouge_scores / length)
    if train_label:
        input, input_lengths = encoding(input_list, in_tk2id, max_length=max_input)
        summary, summary_lengths = encoding(output_list, out_tk2id, max_length=max_output)
        vld_input, vld_input_lengths = encoding(validation_input_list, in_tk2id, max_length=max_input)
        vld_summary, vld_summary_lengths = encoding(validation_output_list, in_tk2id, max_length=max_output)
        print('Data prepared')
        n_iters = 10
        # quick_rouge(encoder1, attn_decoder1, test_input_list, test_output_list, in_tk2id, out_lang)
        print('Start training')
        for i in range(n_iters):
            if epoch == 20:
                print('lr:', learning_rate)
                encoder_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
                decoder_optimizer = optim.Adam(attn_decoder1.parameters(), lr=learning_rate)
                print('Optimizer Initialized')
                # validation(encoder1, attn_decoder1, encoder_optimizer, decoder_optimizer, steps, batch_size,
                #            vld_input, vld_input_lengths, vld_summary, vld_summary_lengths,
                #            print_every=print_every, device=device, max_target_length=max_output)
                # validation(encoder1, attn_decoder1, encoder_optimizer, decoder_optimizer, steps, batch_size,
                #            vld_input, vld_input_lengths, vld_summary, vld_summary_lengths,
                #            print_every=print_every, device=device, max_target_length=100)
            trainIters(encoder1, attn_decoder1, encoder_optimizer, decoder_optimizer, steps, batch_size, input, input_lengths, summary, summary_lengths,
                       print_every=print_every, device=device, max_target_length=max_output, loss_type='rl', opt={'clip':0.2})
            # validation(encoder1, attn_decoder1, encoder_optimizer, decoder_optimizer, steps, batch_size,
            #            vld_input, vld_input_lengths, vld_summary, vld_summary_lengths,
            #            print_every=print_every, device=device, max_target_length=max_output)
            quick_rouge(encoder1, attn_decoder1, test_input_list, test_output_list, in_tk2id, out_lang)
            ckpid = epoch % 10
            checkpoint_name = ckp_path + rl_checkpoint_name_base + '_%d.tar' %ckpid
            if not test:
                torch.save({
                'encoder_state_dict': encoder1.state_dict(),
                'decoder_state_dict': attn_decoder1.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'epoch': epoch
                }, checkpoint_name)
            epoch += 1
    if test:
        input, input_lengths = encoding(input_list, in_tk2id, max_length=max_input)
        length = len(input)
        rouge_scores = 0
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i in range(10):
            sent = input[i]
            true = output_list[i]
            input_length = [input_lengths[i]]
            # prediction,_ = evaluate(encoder1, attn_decoder1,  sent, input_length, out_lang, max_length=MAX_LENGTH)
            prediction = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=3, min_target_length=0, block_ngram_repeat=3, printnum=1)
            print('prediction: ' + ' '.join(prediction))
            print('true: ' +  ' '.join(true))
            scores = scorer.score(' '.join(true), ' '.join(prediction))
            rouge_scores += scores['rougeL'][2]
        print('Rouge-L', rouge_scores/10)
    if (not test and inference):
        test_input, test_input_lengths = encoding(test_input_list, in_tk2id, max_length=max_input)
        length = len(test_input)
        #length = 1000
        # test_filename = 'testout/test-bl.txt'
        # gold_truth = 'testout/gold.txt'
        f1 = open(test_filename, 'w', encoding='utf-8')
        f2 = open(gold_truth, 'w', encoding='utf-8')
        rouge_scores = 0
        print(length)
        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i in range(length):
            # if i < 7698:
            #     continue
            sent = test_input[i]
            true = test_output_list[i]
            input_length = [test_input_lengths[i]]
            pre_input = test_input_list[i]
            pre_input.append(EOS_token)
            #prediction,_ = evaluate(encoder1, attn_decoder1,  sent, input_length, out_lang, max_length=MAX_LENGTH)
            prediction = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=5, min_target_length=5,
                                           printnum=1, block_ngram_repeat=3, max_target_length=max_output, pre_input=pre_input)
            # prediction_2 = beam_evaluate_unk(encoder1, attn_decoder1, sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=5, min_target_length=20, block_ngram_repeat=0, printnum=1)
            if prediction[-1] == 'EOS':
                prediction.pop()
            else:
                pass
                #print(len(prediction))
            prediction_new = [i for i in prediction if not (i == '<t>')]
            true_new = [i for i in true if not (i == '<t>')]
            if prediction_new == []:
                prediction_new = ['UNK']
            scores = scorer.score(' '.join(true_new), ' '.join(prediction_new))
            rouge_scores += scores['rougeL'][2]
            # prediction_new_2 = [i for i in prediction_2 if not (i == '<t>')]
            # print(' '.join(prediction_new) + '\n')
            # print(' '.join(prediction_new_2) + '\n')
            # print(' '.join(true_new) + '\n')
            f1.write(' '.join(prediction_new) + '\n')
            f2.write(' '.join(true_new) + '\n')
            # print('Prediction: ')
            # print(' '.join(prediction))
            # print('Gold truth: ')
            # print(' '.join(true))
        print('Rouge-L', rouge_scores / length)

        # gold_dir = './gold_sum/'
        # pred_dir = './my_sum/'
        # length = len(test_input)
        # for i in range(length):
        #     sent = test_input[i]
        #     true = test_output_list[i]
        #     input_length = [test_input_lengths[i]]
        #     prediction = beam_evaluate_unk(encoder1, attn_decoder1,  sent, input_length, out_lang, max_length=MAX_LENGTH, beam_size=5)
        #     if prediction[-1] == 'EOS':
        #         prediction.pop()
        #     pred = ' '.join(prediction)
        #     gold = ' '.join(true)
        #     #gold_sent = sent_tokenize(gold)
        #     #pred_sent = sent_tokenize(pred)
        #     gold_file = 'gold.A.%05d.txt' %(i+1)
        #     pred_file = 'pred.%05d.txt' %(i+1)
        #     with open(gold_dir + gold_file, 'w', encoding='utf-8') as f:
        #         #f.write('\n'.join(gold_sent))
        #         f.write(gold)
        #     with open(pred_dir + pred_file, 'w', encoding='utf-8') as g:
        #         #g.write('\n'.join(pred_sent))
        #         g.write(pred)
        # r = Rouge155()
        # r.system_dir = pred_dir
        # r.model_dir = gold_dir
        # r.system_filename_pattern = 'pred.(\d+).txt'
        # r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'
        # output = r.convert_and_evaluate()
        # print(output)
        # output_dict = r.output_to_dict(output)

