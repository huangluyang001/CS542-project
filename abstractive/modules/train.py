from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time, math
#import preprocess4model
#import pyrouge
from modules.masked_cross_entropy import masked_cross_entropy
from torch.nn import functional
from rouge.rouge_scorer import RougeScorer
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


class Lang:
    def __init__(self, name, word2index, index2word):
        self.name = name
        self.word2index = word2index
        self.word2count = {}
        self.index2word = index2word
        self.n_words = len(word2index)


PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 400
MAX_TARGET_LENGTH = 100


def pack_seq(seq_list):
    return torch.cat([_.unsqueeze(1) for _ in seq_list], 1)

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    length_of_dataset = len(batch[0])
    return tuple([torch.stack([_[i] for _ in batch]) for i in range(length_of_dataset)])


def gen_dataloader(opt, batch_size):
    if len(opt) == 4:
        dataset = TensorDataset(opt[0], opt[1], opt[2], opt[3])
    elif len(opt) == 4:
        dataset = TensorDataset(opt[0], opt[1], opt[2], opt[3], opt[4])
    else:
        raise Exception('Not supported right now.')
    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, drop_last=True)
    return dataloader

def sample(decoder, encoder_hidden, encoder_outputs, max_target_length, batch_size, target_batches,
           device, sample_max, min_length=0):
    temperature = 1
    seq = []  # baseline, greedy encoding
    seqLogProbs = []
    if sample_max == 1:
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device, dtype=torch.long).detach()
        h_size = (1, batch_size, decoder.hidden_size)
        input_feed = encoder_hidden[0].data.new(*h_size).zero_().detach()
    else:
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device, dtype=torch.long).detach()
        h_size = (1, batch_size, decoder.hidden_size)
        input_feed = encoder_hidden[0].data.new(*h_size).zero_()
    for t in range(max_target_length):
        if sample_max == 1:
            with torch.no_grad():
                decoder_output, decoder_hidden, decoder_attn, input_feed = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, input_feed
                )
        else:
            decoder_output, decoder_hidden, decoder_attn, input_feed = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_feed
            )
        decoder_output = functional.log_softmax(decoder_output, dim=-1).squeeze(0)
        if sample_max == 1: # greedy
            sampleProb, it = torch.max(decoder_output.data, 1)  # Next Input is current prediction
            it = it.detach()
            sampleProb = sampleProb.detach()
        else: # sample
            prob_prev = torch.exp(decoder_output / temperature)
            it = torch.multinomial(prob_prev, 1).detach()# Next Input is current target
            sampleProb = decoder_output.gather(1, it)
            # print(prob_prev.gather(1, it))
            # print(it, sampleProb)
        decoder_input = it
        #decoder_input = target_batches[:, t]
        it = it.view(-1).long()
        if t == 0:
            unfinished = (it != EOS_token)
        else:
            if t + 1 > min_length - 1:
                it = it * unfinished.type_as(it)
                unfinished = unfinished * (it != EOS_token)
        seq.append(it)
        seqLogProbs.append(sampleProb)
        if unfinished.data.sum() == 0:
            break
    return seq, seqLogProbs


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, max_target_length = MAX_TARGET_LENGTH, device='cuda'
          , loss_type='ml', opt = {}):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # seq x batch
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3
    clip = opt.get('clip', 2)
    sample_times = opt.get('sample_times', 5)
    stemmer_id = opt.get('stemmer_id', [EOS_token, SOS_token, 4, 5])

    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    min_length = opt.get('min_length', 0)
    batch_size = input_batches.size(0)

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

    loss = 0

    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device, dtype=torch.long)
    decoder_hidden = encoder_hidden
    h_size = (1, batch_size, decoder.hidden_size)
    input_feed = encoder_hidden[0].data.new(*h_size).zero_()

    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).to(device)
    if loss_type == 'ml':
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, input_feed = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_feed
            )
            all_decoder_outputs[t,:,:] = decoder_output
            decoder_input = target_batches[:,t] # Next Input is current target

        all_decoder_outputs = all_decoder_outputs.permute(1, 2, 0).to(device)
        # print(all_decoder_outputs.size())
        # padding should be masked
        loss = masked_cross_entropy(all_decoder_outputs.permute(0,2,1).contiguous(), target_batches.contiguous(), target_lengths)
        print_loss = loss.item()
    elif loss_type == 'rl' or loss_type == 'ml+rl':
        # baseline
        detached_hidden = (encoder_hidden[0].detach(), encoder_hidden[1].detach())
        bl_seq, seqLogProbs_bl = sample(decoder, detached_hidden, encoder_outputs.detach(), max_target_length, batch_size, target_batches,
                                        device, sample_max=1, min_length=min_length)

        sample_seq, seqLogProbs_sample = sample(decoder, encoder_hidden, encoder_outputs, max_target_length, batch_size, target_batches, device,
                                        sample_max=0, min_length=min_length)

        #print("Greedy Loss:", loss_greedy.item())

        bl_seqs = pack_seq(bl_seq).detach().tolist()
        bl_seqLogProb = pack_seq(seqLogProbs_bl)


        sp_seqs = pack_seq(sample_seq)
        _masks = (sp_seqs > 0).float()
        sp_seqs = sp_seqs.detach().tolist()
        sp_seqLogProb = pack_seq(seqLogProbs_sample)



        loss_nll = - sp_seqLogProb.squeeze(2) * _masks.detach().type_as(sp_seqLogProb)
        bl_scores = []
        tgt_scores = []
        for bl_seq, sp_seq, tgt, length in zip(bl_seqs, sp_seqs, target_batches.tolist(), target_lengths):
            try:
                stop_num = bl_seq.index(EOS_token) + 1
            except ValueError:
                stop_num = max_target_length + 1
            if stop_num < min_length:
                stop_num = min_length
            bl_text = ' '.join([str(_) for _ in bl_seq[:stop_num] if _ not in stemmer_id])
            try:
                stop_num = sp_seq.index(EOS_token) + 1
            except ValueError:
                stop_num = max_target_length + 1
            if stop_num < min_length:
                stop_num = min_length
            sp_text = ' '.join([str(_) for _ in sp_seq[:stop_num] if _ not in stemmer_id])
            target = ' '.join([str(_) for _ in tgt[:length] if _ not in stemmer_id])
            # print ('baseline:', bl_text)
            # print('sample:', sp_text)
            # print('target:', target)
            scores = scorer.score(target, bl_text)
            bl_score = scores['rouge1'][2]
            scores = scorer.score(target, sp_text)
            sp_score = scores['rouge1'][2]
            bl_scores.append(bl_score)
            tgt_scores.append(sp_score)
            # print('baseline:', bl_score)
            # print("sample:", sp_score)
        #print(bl_text, sp_text, target)
        bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=device)
        tgt_scores = torch.tensor(tgt_scores, dtype=torch.float32, device=device)
        #factor = torch.clamp((tgt_scores.view(-1, 1) - bl_scores.view(-1, 1)), 0, 100)
        reward_accum = 0
        reward = tgt_scores.view(-1, 1) - bl_scores.view(-1, 1)
        reward_accum += reward.mean().item()
        # print(reward)


        reward.requires_grad_(False)
        #reward = torch.clamp(reward, min=0, max=1)
        # print(reward)
        loss = reward.detach() * loss_nll
        eps = 1e-10
        #full length = ((_masks * reward) > 0).data.float().sum() + eps
        full_length = _masks.data.float().sum()
        loss = loss.sum() / full_length

        if sample_times > 1:
            for sample_time in range(sample_times-1):
                sample_seq_2, seqLogProbs_sample_2 = sample(decoder, encoder_hidden, encoder_outputs, max_target_length,
                                                        batch_size, target_batches, device,
                                                        sample_max=0, min_length=min_length)
                sp_seqs_2 = pack_seq(sample_seq_2)
                _masks_2 = (sp_seqs_2 > 0).float()
                sp_seqs_2 = sp_seqs_2.detach().tolist()
                sp_seqLogProb_2 = pack_seq(seqLogProbs_sample_2)
                loss_nll_2 = - sp_seqLogProb_2.squeeze(2) * _masks_2.detach().type_as(sp_seqLogProb_2)
                tgt_scores_2 = []
                for bl_seq, sp_seq_2, tgt, length in zip(bl_seqs, sp_seqs_2, target_batches.tolist(), target_lengths):
                    try:
                        stop_num = sp_seq_2.index(EOS_token) + 1
                    except ValueError:
                        stop_num = max_target_length + 1
                    if stop_num < min_length:
                        stop_num = min_length
                    sp_text_2 = ' '.join([str(_) for _ in sp_seq_2[:stop_num] if _ not in stemmer_id])
                    target = ' '.join([str(_) for _ in tgt[:length] if _ not in stemmer_id])
                    scores = scorer.score(target, sp_text_2)
                    sp_score_2 = scores['rouge1'][2]
                    tgt_scores_2.append(sp_score_2)
                tgt_scores_2 = torch.tensor(tgt_scores_2, dtype=torch.float32, device=device)
                reward_2 = tgt_scores_2.view(-1, 1) - bl_scores.view(-1, 1)
                reward_accum += reward_2.mean().item()
                reward_2.requires_grad_(False)
                loss_2 = reward_2.detach() * loss_nll_2
                loss += loss_2.sum() / _masks_2.data.float().sum()
            loss = loss / sample_times
            reward_accum = reward_accum / sample_times
        # print('average greedy rouge: ' + str(round(bl_scores.mean().item(), 4)) + ' average sample reward: ' + str(
        #     round(reward_accum, 4)))
        #loss = loss.sum()
        #print_loss = loss_nll.sum() / length.float().sum()
        print_loss = reward_accum



    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    decoder_optimizer.step()
    encoder_optimizer.step()

    return loss.item(), print_loss

def valid(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,  encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, max_target_length = MAX_TARGET_LENGTH, device='cuda'):
    # seq x batch
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3
    clip = 2.0

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        batch_size = input_batches.size(0)

        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)

        loss = 0

        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device, dtype=torch.long)
        #print(decoder_input)
        decoder_hidden = encoder_hidden

        all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).to(device)
        h_size = (1, batch_size, decoder.hidden_size)
        input_feed = encoder_hidden[0].data.new(*h_size).zero_()
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, input_feed = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_feed
            )
            all_decoder_outputs[t,:,:] = decoder_output
            decoder_input = target_batches[:,t] # Next Input is current target

        all_decoder_outputs = all_decoder_outputs.permute(1, 2, 0).to(device)
        # print(all_decoder_outputs.size())
        # padding should be masked
        loss = masked_cross_entropy(all_decoder_outputs.permute(0,2,1).contiguous(), target_batches.contiguous(), target_lengths)

    encoder.train(True)
    decoder.train(True)

    return loss.item()


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

# input_list: a list of article in index form
def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, n_iters, batch_size, input_list,
               input_lengths, output_list, target_lengths, print_every=1000, device='cuda', max_target_length=100,
               loss_type='ml', opt={}):
    start = time.time()

    print_loss_total = 0
    print_ppl_total = 0


    # encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    # decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    criterion = torch.nn.NLLLoss()

    loop = 0
    dataloader = gen_dataloader([torch.tensor(input_list, dtype=torch.long),
                                torch.tensor(output_list, dtype=torch.long),
                                torch.tensor(input_lengths, dtype=torch.long),
                                torch.tensor(target_lengths, dtype=torch.long)],
                                batch_size=batch_size
                                )
    dataIter = enumerate(dataloader)
    # pairs = list(zip(input_list, output_list, input_lengths, target_lengths))
    # sequence = random.sample(range(len(pairs)), len(pairs))

    for iter in range(1, n_iters+1):
        # if len(sequence) == 0:
        #     sequence = random.sample(range(len(pairs)), len(pairs))
        # # random shuffle
        # input_seqs = []
        # target_seqs = []
        # length_seqs = []
        # tglength_seqs = []
        # for ii in range(batch_size):
        #     try:
        #         idx = sequence.pop()
        #         pair = pairs[idx]
        #     except IndexError:
        #         pair = random.choice(pairs)
        #     input_seqs.append(pair[0])
        #     target_seqs.append(pair[1])
        #     length_seqs.append(pair[2])
        #     tglength_seqs.append(pair[3])
        # seq_pairs = sorted(zip(input_seqs, length_seqs, target_seqs, tglength_seqs), key=lambda x:x[1], reverse=True)
        # input_seqs, length_seqs, target_seqs, tglength_seqs = zip(*seq_pairs)
        #input_tensor = input_list[index]
        #target_tensor = output_list[index]
        # input_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
        # target_tensor = torch.tensor(target_seqs, dtype=torch.long, device=device)
        try:
            input_tensor, target_tensor, length_seqs, tglength_seqs = next(dataIter)[1]
            length_seqs = length_seqs.tolist()
            tglength_seqs = tglength_seqs.tolist()
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
        except StopIteration:
            dataIter = enumerate(dataloader)
            input_tensor, target_tensor, length_seqs, tglength_seqs = next(dataIter)[1]
            length_seqs = length_seqs.tolist()
            tglength_seqs = tglength_seqs.tolist()
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
        max_target_length = max(tglength_seqs)
        target_tensor = target_tensor[:, :max_target_length]
        max_input_length = max(length_seqs)
        input_tensor = input_tensor[:, :max_input_length]




        loss, print_loss = train(
             input_tensor, list(length_seqs), target_tensor, list(tglength_seqs), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device=device,
             max_target_length=max_target_length, loss_type = loss_type, opt=opt
            )
        print_loss_total += loss
        print_ppl_total += print_loss
        loop += 1
        if loop % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_ppl_avg = print_ppl_total / print_every
            print_loss_total = 0
            print_ppl_total = 0

            if loss_type == 'ml':
                print('%s (%d %d%%) %.4f' % (timeSince(start, loop / n_iters),
                                         loop, loop / n_iters * 100, print_loss_avg))
            elif loss_type == 'rl':
                print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, loop / n_iters),
                                             loop, loop / n_iters * 100, print_loss_avg, print_ppl_avg))


def validation(encoder, decoder, encoder_optimizer, decoder_optimizer, n_iters, batch_size, input_list, input_lengths, output_list, target_lengths, print_every=1000, device='cuda', max_target_length=100):
    start = time.time()

    print_loss_total = 0


    # encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    # decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    criterion = torch.nn.NLLLoss()

    loop = 0
    pairs = list(zip(input_list, output_list, input_lengths, target_lengths))
    sequence = random.sample(range(len(pairs)), len(pairs))
    n_iters = math.ceil(len(pairs)/batch_size)

    for iter in range(1, n_iters+1):
        if len(sequence) == 0:
            sequence = random.sample(range(len(pairs)), len(pairs))
        # random shuffle
        input_seqs = []
        target_seqs = []
        length_seqs = []
        tglength_seqs = []
        for ii in range(batch_size):
            try:
                idx = sequence.pop()
                pair = pairs[idx]
            except IndexError:
                pair = random.choice(pairs)
            input_seqs.append(pair[0])
            target_seqs.append(pair[1])
            length_seqs.append(pair[2])
            tglength_seqs.append(pair[3])
        seq_pairs = sorted(zip(input_seqs, length_seqs, target_seqs, tglength_seqs), key=lambda x:x[1], reverse=True)
        input_seqs, length_seqs, target_seqs, tglength_seqs = zip(*seq_pairs)
        #input_tensor = input_list[index]
        #target_tensor = output_list[index]
        input_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_seqs, dtype=torch.long, device=device)




        loss = valid(
             input_tensor, list(length_seqs), target_tensor, list(tglength_seqs), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device=device,
             max_target_length=max_target_length
        )
        print_loss_total += loss
    print_loss_avg = print_loss_total / n_iters
    print_loss_total = 0
    print('validation loss', print_loss_avg)



def train_topic(input_batches, input_lengths, tag_seqs, target_batches, target_lengths, encoder, decoder,  encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, max_target_length = MAX_TARGET_LENGTH, device='cuda'):
    # seq x batch
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3
    clip = 2.0


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    batch_size = input_batches.size(0)


    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, tag_seqs)

    loss = 0

    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device, dtype=torch.long)
    #print(decoder_input)
    decoder_hidden = encoder_hidden
    h_size = (1, batch_size, decoder.hidden_size)
    input_feed = encoder_hidden[0].data.new(*h_size).zero_()

    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).to(device)
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn, input_feed = decoder(
            decoder_input, tag_seqs, decoder_hidden, encoder_outputs, input_feed
        )
        all_decoder_outputs[t,:,:] = decoder_output
        decoder_input = target_batches[:,t] # Next Input is current target

    all_decoder_outputs = all_decoder_outputs.permute(1, 2, 0).to(device)
    # print(all_decoder_outputs.size())
    # padding should be masked
    loss = masked_cross_entropy(all_decoder_outputs.permute(0,2,1).contiguous(), target_batches.contiguous(), target_lengths)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    # for param in decoder.parameters():
    #     print(param.grad)



    decoder_optimizer.step()
    encoder_optimizer.step()

    return loss.item()

def valid_topic(input_batches, input_lengths, tag_seqs, target_batches, target_lengths, encoder, decoder,  encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, max_target_length = MAX_TARGET_LENGTH, device='cuda'):
    # seq x batch
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3
    clip = 2.0


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        batch_size = input_batches.size(0)
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, tag_seqs)

        loss = 0

        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device, dtype=torch.long)
        #print(decoder_input)
        decoder_hidden = encoder_hidden
        h_size = (1, batch_size, decoder.hidden_size)
        input_feed = encoder_hidden[0].data.new(*h_size).zero_()

        all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size).to(device)
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn, input_feed = decoder(
                decoder_input, tag_seqs, decoder_hidden, encoder_outputs, input_feed
            )
            all_decoder_outputs[t,:,:] = decoder_output
            decoder_input = target_batches[:,t] # Next Input is current target

        all_decoder_outputs = all_decoder_outputs.permute(1, 2, 0).to(device)
        # print(all_decoder_outputs.size())
        # padding should be masked
        loss = masked_cross_entropy(all_decoder_outputs.permute(0,2,1).contiguous(), target_batches.contiguous(), target_lengths)
        # for param in decoder.parameters():
        #     print(param.grad)
    encoder.train(True)
    decoder.train(True)

    return loss.item()





def trainIters_topic(encoder, decoder, encoder_optimizer, decoder_optimizer, n_iters, batch_size, input_list, input_lengths, tag_list, output_list,
                     target_lengths, print_every=1000, learning_rate=0.15, init_accum=0.1, device='cuda', max_target_length = MAX_TARGET_LENGTH):
    start = time.time()

    print_loss_total = 0


    # encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    # decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    criterion = nn.NLLLoss()

    loop = 0
    pairs = list(zip(input_list, output_list, input_lengths, target_lengths, tag_list))
    sequence = random.sample(range(len(pairs)), len(pairs))

    for iter in range(1, n_iters+1):
        if len(sequence) == 0:
            sequence = random.sample(range(len(pairs)), len(pairs))
        # random shuffle
        input_seqs = []
        target_seqs = []
        length_seqs = []
        tglength_seqs = []
        tag_seqs = []
        for ii in range(batch_size):
            try:
                idx = sequence.pop()
                pair = pairs[idx]
            except IndexError:
                pair = random.choice(pairs)
            input_seqs.append(pair[0])
            target_seqs.append(pair[1])
            length_seqs.append(pair[2])
            tglength_seqs.append(pair[3])
            tag_seqs.append(pair[4])
        seq_pairs = sorted(zip(input_seqs, length_seqs, target_seqs, tglength_seqs, tag_seqs), key=lambda x:x[1], reverse=True)
        input_seqs, length_seqs, target_seqs, tglength_seqs, tag_seqs = zip(*seq_pairs)
        #input_tensor = input_list[index]
        #target_tensor = output_list[index]
        input_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_seqs, dtype=torch.long, device=device)
        tag_tensor = torch.tensor(tag_seqs, dtype=torch.float, device=device)




        loss = train_topic(
             input_tensor, list(length_seqs), tag_tensor, target_tensor, list(tglength_seqs),
            encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device=device,
            max_target_length=max_target_length
        )
        print_loss_total += loss
        loop += 1
        if loop % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print('%s (%d %d%%) %.4f' % (timeSince(start, loop / n_iters),
                                         loop, loop / n_iters * 100, print_loss_avg))


def validation_topic(encoder, decoder, encoder_optimizer, decoder_optimizer, n_iters, batch_size, input_list, input_lengths, tag_list, output_list, target_lengths, print_every=1000, learning_rate=0.15, init_accum=0.1, device='cuda'):
    start = time.time()

    print_loss_total = 0


    # encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    # decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, initial_accumulator_value=init_accum)
    criterion = nn.NLLLoss()

    loop = 0
    pairs = list(zip(input_list, output_list, input_lengths, target_lengths, tag_list))
    sequence = random.sample(range(len(pairs)), len(pairs))
    n_iters = math.floor(len(pairs) / batch_size)

    for iter in range(1, n_iters+1):
        if len(sequence) == 0:
            sequence = random.sample(range(len(pairs)), len(pairs))
        # random shuffle
        input_seqs = []
        target_seqs = []
        length_seqs = []
        tglength_seqs = []
        tag_seqs = []
        for ii in range(batch_size):
            try:
                idx = sequence.pop()
                pair = pairs[idx]
            except IndexError:
                pair = random.choice(pairs)
            input_seqs.append(pair[0])
            target_seqs.append(pair[1])
            length_seqs.append(pair[2])
            tglength_seqs.append(pair[3])
            tag_seqs.append(pair[4])
        seq_pairs = sorted(zip(input_seqs, length_seqs, target_seqs, tglength_seqs, tag_seqs), key=lambda x:x[1], reverse=True)
        input_seqs, length_seqs, target_seqs, tglength_seqs, tag_seqs = zip(*seq_pairs)
        #input_tensor = input_list[index]
        #target_tensor = output_list[index]
        input_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)
        target_tensor = torch.tensor(target_seqs, dtype=torch.long, device=device)
        tag_tensor = torch.tensor(tag_seqs, dtype=torch.float, device=device)





        loss = train_topic(
             input_tensor, list(length_seqs), tag_tensor, target_tensor, list(tglength_seqs), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device=device
        )
        print_loss_total += loss
    print_loss_avg = print_loss_total / n_iters
    #print('%s (%d %d%%) %.4f' % (timeSince(start, loop / n_iters),
    #                                 loop, loop / n_iters * 100, print_loss_avg))
    print('validation loss', print_loss_avg)

