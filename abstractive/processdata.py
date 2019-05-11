import json
import glob
import collections, pickle

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

path = 'D:\\542pj\data\\train'
files = glob.glob(path + '\\*')
input_list = []
sum_list = []
vocab_counter = collections.Counter()
i = 0
for file in files:
    f = open(file)
    data = json.loads(f.read())
    _input = data['article']
    _abstract = data['abstract']
    _input = ' '.join(_input).lower().split(' ')
    _abstract = ' '.join(_abstract).lower().split(' ')
    _input = [_ for _ in _input if _ != '']
    _abstract = [_ for _ in _abstract if _ != ""]
    input_list.append(_input)
    sum_list.append(_abstract)
    vocab_counter.update(_input)
    vocab_counter.update(_abstract)
    i += 1
    if i % 10000 == 0:
        print('10000 finished')

print(sorted(vocab_counter.items(), key=lambda x:x[1], reverse=True)[:50000])
vocab = sorted(vocab_counter.items(), key=lambda x:x[1], reverse=True)[:50000]
word2id = {
    'PAD':0,
    'SOS':1,
    'EOS':2,
    'UNK':3
}
_id = 4
for word, count in vocab:
    word2id.setdefault(word, _id)
    _id += 1
print('word2id:', word2id['he'], len(word2id))
id2word = {}
for word, _id in word2id.items():
    id2word.setdefault(_id, word)

train_data = [word2id, id2word, word2id, id2word, input_list, sum_list]
with open('train.pkl', 'wb') as f:
    pickle.dump(train_data, f)

print(input_list[0], sum_list[0])

with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)
word2id, id2word, _, _, _, _ = train_data

path = 'D:\\542pj\data\\val'
files = glob.glob(path + '\\*')
input_list = []
sum_list = []
for file in files:
    try:
        f = open(file)
        data = json.loads(f.read())
    except:
        print(file)
    _input = data['article']
    _abstract = data['abstract']
    _input = ' '.join(_input).lower().split(' ')
    _abstract = ' '.join(_abstract).lower().split(' ')
    _input = [_ for _ in _input if _ != '']
    _abstract = [_ for _ in _abstract if _ != ""]
    input_list.append(_input)
    sum_list.append(_abstract)

val_data = [word2id, id2word, word2id, id2word, input_list, sum_list]
with open('val.pkl', 'wb') as f:
    pickle.dump(val_data, f)

path = 'D:\\542pj\data\\test'
files = glob.glob(path + '\\*')
input_list = []
sum_list = []
for file in files:
    f = open(file)
    data = json.loads(f.read())
    _input = data['article']
    _abstract = data['abstract']
    _input = ' '.join(_input).lower().split(' ')
    _abstract = ' '.join(_abstract).lower().split(' ')
    _input = [_ for _ in _input if _ != '']
    _abstract = [_ for _ in _abstract if _ != ""]
    input_list.append(_input)
    sum_list.append(_abstract)
test_data = [word2id, id2word, word2id, id2word, input_list, sum_list]
with open('test.pkl', 'wb') as f:
    pickle.dump(test_data, f)


