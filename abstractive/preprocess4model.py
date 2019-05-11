import pickle
import numpy as np
import random
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import word_tokenize
import re
import math
from pycorenlp import StanfordCoreNLP

tag2id = {
'science': 0,
'government': 1,
'arts': 2,
'law': 3,
'business': 4,
'sports': 5,
'lifestyle': 6,
'education': 7,
'health': 8,
'technology': 9,
'military': 10,

}











class Preprocessing():
    def __init__(self):
        self.data_folder =  '/data/luyang/'
        self.nyt_file = 'nyt_data_processed_new.pkl'
        self.filtered_nyt_file = 'nyt_data_processed_filtered.pkl'
        self.filtered_nyt_file2 = 'nyt_data_processed_filtered2.pkl'
        pass

    @staticmethod
    def dict_info(sample_dict):
        count_dict = {}
        count_single_dict = {}
        count_double_dict = {}
        count_triple_dict = {}
        for key, data_dict in sample_dict.items():
            tags = data_dict['tag']
            try:
                count_dict[len(tags)] += 1
            except KeyError:
                count_dict.setdefault(len(tags), 1)
            if len(tags) == 1:
                try:
                    count_single_dict[list(tags)[0]] += 1
                except KeyError:
                    count_single_dict.setdefault(tags[0],1)
            if len(tags) == 2:
                taglist = list(tags)
                taglist = sorted(taglist)
                try:
                    count_double_dict[taglist[0]+','+taglist[1]] += 1
                except KeyError:
                    count_double_dict.setdefault(list(tags)[0]+','+list(tags)[1],1)

            if len(tags) == 3:
                taglist = list(tags)
                taglist = sorted(taglist)
                try:
                    count_triple_dict[taglist[0]+','+taglist[1]+','+taglist[2]] += 1
                except KeyError:
                    count_triple_dict.setdefault(taglist[0]+','+taglist[1]+','+taglist[2],1)
        print('| topic | count |')
        print('| ----- | ----- |')
        for key, value in sorted(count_dict.items(),key=lambda x:x[1], reverse=True):
            print('|' + str(key) + '|' + str(value) + '|')
        print('Tag Distribution')
        print('| topic pairs | count |')
        print('| ----- | ----- |')
        for key, value in sorted(count_single_dict.items(),key=lambda x:x[1], reverse=True):
            print('|' + str(key) + '|' + str(value) + '|')
        print('Two tags')
        print('| topic pairs | count |')
        print('| ----- | ----- |')
        for key, value in sorted(count_double_dict.items(),key=lambda x:x[1], reverse=True):
            print('|' + str(key)+'|'+str(value) + '|')
        print('Three tags')
        print('| topic pairs | count |')
        print('| ----- | ----- |')
        i =0
        for key, value in sorted(count_triple_dict.items(),key=lambda x:x[1], reverse=True):
            print('|'+str(key)+'|'+str(value)+'|')
            i += 1
            if i > 49:
                break

    def filter_zero_tag(self):
        with open(self.data_folder+self.nyt_file, 'rb') as f:
            nyt_dict = pickle.load(f)
        count = 0
        new_dict = {}
        for key, news_dict in nyt_dict.items():
            tags = news_dict['tag']
            if len(tags) == 0:
                count += 1
            else:
                new_dict.setdefault(key, news_dict)
        with open(self.data_folder+self.filtered_nyt_file2, 'wb') as f:
            pickle.dump(new_dict, f)
        print(len(new_dict))
        print(count)


    def train_test_split(self):
        with open(self.data_folder+self.filtered_nyt_file2, 'rb') as f:
            nyt_dict = pickle.load(f)
        print('Total articles',len(nyt_dict))
        random.seed(0)
        sequence = random.sample(range(len(nyt_dict)),len(nyt_dict))
        print(sequence[0:5])
        test_sequence = sequence[:math.floor(len(nyt_dict)/10)]
        development_sequence = sequence[math.floor(len(nyt_dict)/10):math.floor(len(nyt_dict)/5)]
        train_sequence = sequence[math.floor(len(nyt_dict)/5):]
        print(len(test_sequence))
        print(len(train_sequence))
        print(len(development_sequence))
        train_dict = {}
        test_dict = {}
        development_dict = {}
        i = 0
        nyt_list = list(nyt_dict.items())
        print(len(nyt_list))
        train_dict = dict([nyt_list[i] for i in train_sequence])
        test_dict = dict([nyt_list[i] for i in test_sequence])
        development_dict = dict([nyt_list[i] for i in development_sequence])


        print('# of Development Data:',len(development_dict))
        print('# of Test Data:', len(test_dict))
        print('# of Train Data:', len(train_dict))
        print('Train info:')
        self.dict_info(train_dict)
        print('Test info:')
        self.dict_info(test_dict)
        print('Development info:')
        self.dict_info(development_dict)
        with open('train_nyt.pkl','wb') as f:
            pickle.dump(train_dict,f)
        with open('test_nyt.pkl','wb') as f:
            pickle.dump(test_dict,f)
        with open('dvp_nyt.pkl','wb') as f:
            pickle.dump(development_dict,f)

    def filter(self, max_len_in = 400, max_len_sum = 100, intokennum=150000, outtokennum=50000):
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        #tokenizer = StanfordTokenizer()
        with open(self.data_folder+self.nyt_file,'rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        input_length = 0
        output_length = 0
        ly_i = 0
        word_count = {}
        new_train_dict = {}
        for key, news_dict in train_dict.items():
            abstract = ''.join(news_dict['abstract']).lower()
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            summary = word_tokenize(abstract)
            if summary[-1] in ban_list and summary[-2] == ';':
                summary.pop()
                summary.pop()
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            norm_bl = word_tokenize(''.join(norm_bl.split(',')))
            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            paragraphs = paragraphs.lower()
            title = title.lower()
            input = [title_token] + word_tokenize(title) + [byline_token] + word_tokenize(byline) + [content_token] + word_tokenize(paragraphs)
            if len(input) > 100 and len(summary) > 20:
                if len(input) > max_len_in:
                    input = input[:max_len_in]
                if len(summary) > max_len_sum:
                    summary = summary[:max_len_sum]
                #input.append('EOS')
                #summary.append('EOS')
                input_list.append(input)
                output_list.append(summary)
                input_length += len(input)
                output_length += len(summary)
                ly_i += 1
                for word in input + summary:
                    try:
                        word_count[word] += 1
                    except KeyError:
                        word_count.setdefault(word, 1)
                new_train_dict.setdefault(key, news_dict)
        print('All number, ', len(new_train_dict))
        with open(self.data_folder+'nyt_data_processed_filtered.pkl','wb') as f:
            pickle.dump(new_train_dict,f)
        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        def make_dict(word_count, tokennum):
            if tokennum < len(word_count):
                input_word = sorted(word_count.items(),key=lambda x:x[1],reverse=True)[:tokennum]
            else:
                input_word = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            in_tkset = [token for token,_ in input_word]
            in_tkset.insert(0, 'UNK')
            in_tkset.insert(0, 'EOS')
            in_tkset.insert(0, 'SOS')
            in_tkset.insert(0, 'PAD')
            in_tk2id = {in_tkset[i]:i for i in range(len(in_tkset))}
            in_id2tk = {i:in_tkset[i] for i in range(len(in_tkset))}
            return in_tk2id, in_id2tk
        in_tk2id, in_id2tk = make_dict(word_count, intokennum)
        out_tk2id, out_id2tk = make_dict(word_count, outtokennum)

        print('Number of intokens: ', len(in_tk2id))
        print('Number of outtokens: ', len(out_tk2id))
        print('Number of intokens: ', len(in_id2tk))
        print('Number of outtokens: ', len(out_id2tk))
        #print(in_id2tk)
        # print(tk2id)
        return in_tk2id,in_id2tk,out_tk2id,out_id2tk,input_list,output_list

    @staticmethod
    def find_author(input, nm_bl, author_token):
        bl_list = []
        id_list = []
        j = 0
        new_input = []
        substitute = 0
        for i in range(len(input)):
            word = input[i]
            if word in nm_bl:
                if i == j+1:
                    id_list.append(i)
                    j = i
                elif len(id_list) == 0:
                    id_list.append(i)
                    j = i
                else:
                    id_list = []
            elif len(id_list) > 1:
                for j in range(len(id_list)):
                    new_input.pop()
                new_input.append(author_token)
                id_list = []
                substitute = 1

            new_input.append(word)
        return new_input, substitute




    def train_preprocess_mini(self, max_len_in = 400-1, max_len_sum = 100-1, intokennum=150000, outtokennum=50000, write_file=False):
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        with open('train_nyt.pkl','rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        input_length = 0
        output_length = 0
        ly_i = 0
        word_count = {}
        for key, news_dict in train_dict.items():
            abstract = ''.join(news_dict['abstract']).lower()
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            summary = word_tokenize(abstract)
            summary_raw = word_tokenize(abstract)
            if summary[-1] in ban_list and summary[-2] == ';':
                summary.pop()
                summary.pop()
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            norm_bl = word_tokenize(''.join(norm_bl.split(',')))
            bl_token = word_tokenize(byline)
            for i in range(len(bl_token)):
                if bl_token[i] == 'by':
                    try:
                        norm_bl.append(bl_token[i+1])
                    except IndexError:
                        continue
                    try:
                        norm_bl.append(bl_token[i+2])
                    except IndexError:
                        continue
                    try:
                        if bl_token[i+3] != '(':
                            norm_bl.append(bl_token[i+3])
                    except IndexError:
                        continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            input_raw = [title_token] + word_tokenize(title) + [byline_token] + word_tokenize(byline) + [content_token] + word_tokenize(paragraphs)
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            paragraphs = paragraphs.lower()
            title = title.lower()
            input = [title_token] + word_tokenize(title) + [content_token] + word_tokenize(paragraphs)
            if len(input) > 100 and len(summary) > 20:
                input, _ = self.find_author(input, norm_bl, author_token)
                summary, substitute = self.find_author(summary, norm_bl, author_token)
                # print('input: ', input)
                # print('summary: ', summary)
                # print('byline:', norm_bl)
                if len(input) > max_len_in:
                    input = input[:max_len_in]
                if len(summary) > max_len_sum:
                    summary = summary[:max_len_sum]
                #input.append('EOS')
                #summary.append('EOS')
                input_list.append(input)
                output_list.append(summary)
                input_length += len(input)
                output_length += len(summary)
                for word in input + summary:
                    try:
                        word_count[word] += 1
                    except KeyError:
                        word_count.setdefault(word, 1)
                ly_i += 1
                if ly_i > 9:
                    break
                data_folfer = '/data/luyang/luyangdata/nyt/sampleinput/'
                file_name = 'sampleinput_all'  + '.txt'
                if write_file == True and substitute == 1:
                    with open(data_folfer+file_name, 'a+', encoding='utf-8') as f:
                        f.write('--------------Raw-Input---------------\n')
                        f.write(' '.join(input_raw))
                        f.write('\n\n')
                        f.write('--------------Input---------------\n')
                        f.write(' '.join(input))
                        f.write('\n\n')
                        f.write('--------------Raw-Summary---------------\n')
                        f.write(' '.join(summary_raw))
                        f.write('\n\n')
                        f.write('--------------------Summary----------------\n')
                        f.write(' '.join(summary))
                        f.write('\n\n\n\n\n')


        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        def make_dict(word_count, tokennum):
            if tokennum < len(word_count):
                input_word = sorted(word_count.items(),key=lambda x:x[1],reverse=True)[:tokennum]
            else:
                input_word = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            in_tkset = [token for token,_ in input_word]
            in_tkset.insert(0, 'UNK')
            in_tkset.insert(0, 'EOS')
            in_tkset.insert(0, 'SOS')
            in_tkset.insert(0, 'PAD')
            in_tk2id = {in_tkset[i]:i for i in range(len(in_tkset))}
            in_id2tk = {i:in_tkset[i] for i in range(len(in_tkset))}
            return in_tk2id, in_id2tk
        in_tk2id, in_id2tk = make_dict(word_count, intokennum)
        out_tk2id, out_id2tk = make_dict(word_count, outtokennum)

        print('Number of intokens: ', len(in_tk2id))
        print('Number of outtokens: ', len(out_tk2id))
        print('Number of intokens: ', len(in_id2tk))
        print('Number of outtokens: ', len(out_id2tk))
        #print(in_id2tk)
        # print(tk2id)
        return in_tk2id,in_id2tk,out_tk2id,out_id2tk,input_list,output_list

    def train_preprocess_all(self, max_len_in = 400-1, max_len_sum = 100-1, intokennum=150000, outtokennum=50000, write_file=False):
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        with open('train_nyt.pkl','rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        input_length = 0
        output_length = 0
        ly_i = 0
        word_count = {}
        for key, news_dict in train_dict.items():
            abstract = ''.join(news_dict['abstract']).lower()
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            summary = word_tokenize(abstract)
            summary_raw = word_tokenize(abstract)
            if summary[-1] in ban_list and summary[-2] == ';':
                summary.pop()
                summary.pop()
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            norm_bl = word_tokenize(''.join(norm_bl.split(',')))
            bl_token = word_tokenize(byline)
            for i in range(len(bl_token)):
                if bl_token[i] == 'by':
                    try:
                        norm_bl.append(bl_token[i+1])
                    except IndexError:
                        continue
                    try:
                        norm_bl.append(bl_token[i+2])
                    except IndexError:
                        continue
                    try:
                        if bl_token[i+3] != '(':
                            norm_bl.append(bl_token[i+3])
                    except IndexError:
                        continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            input_raw = [title_token] + word_tokenize(title) + [byline_token] + word_tokenize(byline) + [content_token] + word_tokenize(paragraphs)
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            paragraphs = paragraphs.lower()
            title = title.lower()
            input = [title_token] + word_tokenize(title) + [byline_token] + word_tokenize(byline) + [content_token] + word_tokenize(paragraphs)
            if len(input) > 100 and len(summary) > 20:
                input, _ = self.find_author(input, norm_bl, author_token)
                summary, substitute = self.find_author(summary, norm_bl, author_token)
                if len(input) > max_len_in:
                    input = input[:max_len_in]
                if len(summary) > max_len_sum:
                    summary = summary[:max_len_sum]
                #input.append('EOS')
                #summary.append('EOS')
                input_list.append(input)
                output_list.append(summary)
                input_length += len(input)
                output_length += len(summary)
                for word in input + summary:
                    try:
                        word_count[word] += 1
                    except KeyError:
                        word_count.setdefault(word, 1)


        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        def make_dict(word_count, tokennum):
            if tokennum < len(word_count):
                input_word = sorted(word_count.items(),key=lambda x:x[1],reverse=True)[:tokennum]
            else:
                input_word = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            in_tkset = [token for token,_ in input_word]
            in_tkset.insert(0, 'UNK')
            in_tkset.insert(0, 'EOS')
            in_tkset.insert(0, 'SOS')
            in_tkset.insert(0, 'PAD')
            in_tk2id = {in_tkset[i]:i for i in range(len(in_tkset))}
            in_id2tk = {i:in_tkset[i] for i in range(len(in_tkset))}
            return in_tk2id, in_id2tk
        in_tk2id, in_id2tk = make_dict(word_count, intokennum)
        out_tk2id, out_id2tk = make_dict(word_count, outtokennum)

        print('Number of intokens: ', len(in_tk2id))
        print('Number of outtokens: ', len(out_tk2id))
        print('Number of intokens: ', len(in_id2tk))
        print('Number of outtokens: ', len(out_id2tk))
        #print(in_id2tk)
        # print(tk2id)
        return in_tk2id,in_id2tk,out_tk2id,out_id2tk,input_list,output_list

    def test_preprocess_all(self, max_len_in = 400-1, max_len_sum = 100-1, filename='test_nyt.pkl', numofdata=20):
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        with open(filename,'rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        input_length = 0
        output_length = 0
        ly_i = 0
        word_count = {}
        for key, news_dict in train_dict.items():
            abstract = ''.join(news_dict['abstract']).lower()
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            summary = word_tokenize(abstract)
            summary_raw = word_tokenize(abstract)
            if summary[-1] in ban_list and summary[-2] == ';':
                summary.pop()
                summary.pop()
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            norm_bl = word_tokenize(''.join(norm_bl.split(',')))
            bl_token = word_tokenize(byline)
            for i in range(len(bl_token)):
                if bl_token[i] == 'by':
                    try:
                        norm_bl.append(bl_token[i+1])
                    except IndexError:
                        continue
                    try:
                        norm_bl.append(bl_token[i+2])
                    except IndexError:
                        continue
                    try:
                        if bl_token[i+3] != '(':
                            norm_bl.append(bl_token[i+3])
                    except IndexError:
                        continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            input_raw = [title_token] + word_tokenize(title) + [byline_token] + word_tokenize(byline) + [content_token] + word_tokenize(paragraphs)
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            paragraphs = paragraphs.lower()
            title = title.lower()
            input = [title_token] + word_tokenize(title) + [byline_token] + word_tokenize(byline) + [content_token] + word_tokenize(paragraphs)
            if len(input) > 100 and len(summary) > 20:
                input, _ = self.find_author(input, norm_bl, author_token)
                summary, substitute = self.find_author(summary, norm_bl, author_token)
                if len(input) > max_len_in:
                    input = input[:max_len_in]
                if len(summary) > max_len_sum:
                    summary = summary[:max_len_sum]
                #input.append('EOS')
                #summary.append('EOS')
                input_list.append(input)
                output_list.append(summary)
                input_length += len(input)
                output_length += len(summary)
                ly_i += 1
                if ly_i > numofdata:
                    break
                for word in input + summary:
                    try:
                        word_count[word] += 1
                    except KeyError:
                        word_count.setdefault(word, 1)


        print('test_number: ', len(input_list))
        print('test_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        #print(in_id2tk)
        # print(tk2id)
        return input_list,output_list


    @staticmethod
    def stan_word_tokenize(nlp, text):
        output = nlp.annotate(text, properties={
            'annotators': 'tokenize,ssplit',
            'outputFormat': 'json'
        })
        para_list = []
        try:
            for sentence in output['sentences']:
                sent_list = []
                for word in sentence['tokens']:
                    sent_list.append(word['word'])
                para_list.append(sent_list)
            token_list = [word for sent in para_list for word in sent]
        except TypeError:
            print(text)
            token_list = []
        return para_list, token_list

    @staticmethod
    def tag_onehot(tags):
        tag_list = np.zeros(11)
        for tag in tags:
            tag_list[tag2id[tag]] += 1.
        assert sum(tag_list) > 0.
        return list(tag_list)


    def test_withtopic_all(self, max_len_in = 400-1, max_len_sum = 100-1, intokennum=50000, outtokennum=50000, write_file=False, numoftest=10):
        nlp = StanfordCoreNLP('http://localhost:9000')
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        with open('test_nyt.pkl','rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        tag_list = []
        input_length = 0
        output_length = 0
        ly_i = 0
        word_count = {}
        for key, news_dict in train_dict.items():
            abstract = ''.join(news_dict['abstract']).lower()
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            tags = news_dict['tag']

            _, summary = self.stan_word_tokenize(nlp, abstract)

            #_, summary_raw = self.stan_word_tokenize(nlp, abstract)
            if len(summary) > 5:
                if summary[-1] in ban_list and summary[-2] == ';':
                    summary.pop()
                    summary.pop()
            if len(summary) > 10:
                for i in range(5):
                    if summary[-i-1] in ban_list and summary[-i-2] == ';':
                        for j in range(i+2):
                            summary.pop()
                        break
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            _, norm_bl = self.stan_word_tokenize(nlp, ''.join(norm_bl.split(',')))
            _, bl_token = self.stan_word_tokenize(nlp, byline)

            for i in range(len(bl_token)):
                if bl_token[i] == 'by':
                    try:
                        norm_bl.append(bl_token[i+1])
                    except IndexError:
                        continue
                    try:
                        norm_bl.append(bl_token[i+2])
                    except IndexError:
                        continue
                    try:
                        if bl_token[i+3] != '(':
                            norm_bl.append(bl_token[i+3])
                    except IndexError:
                        continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            paragraphs = paragraphs.lower()
            title = title.lower()
            _, title_list = self.stan_word_tokenize(nlp, title)
            _, para_list = self.stan_word_tokenize(nlp, paragraphs)
            input = [title_token] + title_list + [content_token] + para_list

            input, _ = self.find_author(input, norm_bl, author_token)
            summary, substitute = self.find_author(summary, norm_bl, author_token)
            # print('input: ', input)
            # print('summary: ', summary)
            # print('byline:', norm_bl)
            if len(input) > max_len_in:
                input = input[:max_len_in]
            if len(summary) > max_len_sum:
                summary = summary[:max_len_sum]
            #input.append('EOS')
            #summary.append('EOS')
            input_list.append(input)
            output_list.append(summary)
            tag_list.append(self.tag_onehot(tags))
            input_length += len(input)
            output_length += len(summary)
            for word in input + summary:
                try:
                    word_count[word] += 1
                except KeyError:
                    word_count.setdefault(word, 1)
            ly_i += 1
            if ly_i > numoftest - 1:
                break




        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))

        return input_list,output_list, tag_list


    def make_all_dict(self, max_len_in = 400-1, max_len_sum = 100-1, intokennum=50000, outtokennum=50000,
                            write_file=False, numoftrain=10, filename='train_nyt.pkl',
                            makedict = True, truncated=True):
        nlp = StanfordCoreNLP('http://localhost:9000')
        print('host started')
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        all_data_dict = {}
        with open(filename,'rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        tag_list = []
        input_length = 0
        output_length = 0
        new_all_dict = {}
        for key, news_dict in train_dict.items():
            new_news_dict = {}
            #abstract = ''.join(news_dict['abstract']).lower()
            abstract = ''.join(news_dict['abstract'])
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            abstract = re.sub('\(M\)|\(S\)', '', abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            tags = news_dict['tag']

            headline = news_dict['hdl']
            online_sction = news_dict['online_section']
            date = news_dict['date']
            byline = news_dict['bl']
            nbl = news_dict['normbl']
            gd = news_dict['general_descriptors']
            descriptors = news_dict['descriptors']
            onhdl = news_dict['onhdl']
            new_news_dict.setdefault('hdl', headline)
            new_news_dict.setdefault('online_section', online_sction)
            new_news_dict.setdefault('date', date)
            new_news_dict.setdefault('bl', byline)
            new_news_dict.setdefault('normbl', nbl)
            new_news_dict.setdefault('general_descriptors', gd)
            new_news_dict.setdefault('descriptors', descriptors)
            new_news_dict.setdefault('onhdl', onhdl)
            new_news_dict.setdefault('raw_abstract', news_dict['abstract'])
            new_news_dict.setdefault('raw_paragraphs', news_dict['paragraphs'])



            _, summary = self.stan_word_tokenize(nlp, abstract)
            if summary == []:
                print('-------------summary---------')
                continue

            #_, summary_raw = self.stan_word_tokenize(nlp, abstract)
            if len(summary) > 5:
                if summary[-1] in ban_list and summary[-2] == ';':
                    summary.pop()
                    summary.pop()
            if len(summary) > 10:
                for i in range(5):
                    if summary[-i-1] in ban_list and summary[-i-2] == ';':
                        for j in range(i+2):
                            summary.pop()
                        break
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            _, norm_bl = self.stan_word_tokenize(nlp, ''.join(norm_bl.split(',')))
            _, bl_token = self.stan_word_tokenize(nlp, byline)

            for i in range(len(bl_token)):
                if bl_token[i] == 'by':
                    try:
                        norm_bl.append(bl_token[i+1])
                    except IndexError:
                        continue
                    try:
                        norm_bl.append(bl_token[i+2])
                    except IndexError:
                        continue
                    try:
                        if bl_token[i+3] != '(':
                            norm_bl.append(bl_token[i+3])
                    except IndexError:
                        continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            #paragraphs = paragraphs.lower()
            title = title
            _, title_list = self.stan_word_tokenize(nlp, title)
            _, para_list = self.stan_word_tokenize(nlp, paragraphs)
            if para_list == []:
                print('-------------paragraph---------')
                continue
            input = [title_token] + title_list + [content_token] + para_list

            input, _ = self.find_author(input, norm_bl, author_token)
            summary, substitute = self.find_author(summary, norm_bl, author_token)
            # print('input: ', input)
            # print('summary: ', summary)
            # print('byline:', norm_bl)
            # if truncated == True:
            #     if len(input) > max_len_in:
            #         input = input[:max_len_in]
            #     if len(summary) > max_len_sum:
            #         summary = summary[:max_len_sum]
            #input.append('EOS')
            #summary.append('EOS')
            input_list.append(input)
            output_list.append(summary)
            #tag_list.append(self.tag_onehot(tags))
            input_length += len(input)
            output_length += len(summary)
            new_news_dict.setdefault('input', input)
            new_news_dict.setdefault('summary', summary)
            #new_news_dict.setdefault('topic', self.tag_onehot(tags))
            new_news_dict.setdefault('topic', tags)
            new_all_dict.setdefault(key, new_news_dict)



        print(len(new_all_dict))
        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        return new_all_dict


    def make_all_dict_noparsing(self, max_len_in = 400-1, max_len_sum = 100-1, intokennum=50000, outtokennum=50000,
                            write_file=False, numoftrain=10, filename='train_nyt.pkl',
                            makedict = True, truncated=True):
        nlp = StanfordCoreNLP('http://localhost:9000')
        print('host started')
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        all_data_dict = {}
        with open(filename,'rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        tag_list = []
        input_length = 0
        output_length = 0
        new_all_dict = {}
        for key, news_dict in train_dict.items():
            new_news_dict = {}
            #abstract = ''.join(news_dict['abstract']).lower()
            abstract = ''.join(news_dict['abstract'])
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            abstract = re.sub('\(M\)|\(S\)', '', abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            pattern2 = '; photo.?|; graph.?|; chart.?|; map.?|; table.?|; drawing.?'
            summary = re.sub(pattern2,'',abstract)
            tags = news_dict['tag']

            headline = news_dict['hdl']
            online_sction = news_dict['online_section']
            date = news_dict['date']
            byline = news_dict['bl']
            nbl = news_dict['normbl']
            gd = news_dict['general_descriptors']
            descriptors = news_dict['descriptors']
            onhdl = news_dict['onhdl']
            new_news_dict.setdefault('hdl', headline)
            new_news_dict.setdefault('online_section', online_sction)
            new_news_dict.setdefault('date', date)
            new_news_dict.setdefault('bl', byline)
            new_news_dict.setdefault('normbl', nbl)
            new_news_dict.setdefault('general_descriptors', gd)
            new_news_dict.setdefault('descriptors', descriptors)
            new_news_dict.setdefault('onhdl', onhdl)
            new_news_dict.setdefault('raw_abstract', news_dict['abstract'])
            new_news_dict.setdefault('raw_paragraphs', news_dict['paragraphs'])



            # _, summary = self.stan_word_tokenize(nlp, abstract)
            # if summary == []:
            #     print('-------------summary---------')
            #     continue

            # #_, summary_raw = self.stan_word_tokenize(nlp, abstract)
            # if len(summary) > 5:
            #     if summary[-1] in ban_list and summary[-2] == ';':
            #         summary.pop()
            #         summary.pop()
            # if len(summary) > 10:
            #     for i in range(5):
            #         if summary[-i-1] in ban_list and summary[-i-2] == ';':
            #             for j in range(i+2):
            #                 summary.pop()
            #             break
            title = ''.join(news_dict['hdl'])
            # byline = ''.join(news_dict['bl']).lower()
            # norm_bl = ''.join(news_dict['normbl']).lower()
            # if byline == None:
            #     byline = ''
            # if norm_bl == None:
            #     norm_bl = ''
            #
            # _, norm_bl = self.stan_word_tokenize(nlp, ''.join(norm_bl.split(',')))
            # _, bl_token = self.stan_word_tokenize(nlp, byline)
            #
            # for i in range(len(bl_token)):
            #     if bl_token[i] == 'by':
            #         try:
            #             norm_bl.append(bl_token[i+1])
            #         except IndexError:
            #             continue
            #         try:
            #             norm_bl.append(bl_token[i+2])
            #         except IndexError:
            #             continue
            #         try:
            #             if bl_token[i+3] != '(':
            #                 norm_bl.append(bl_token[i+3])
            #         except IndexError:
            #             continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            #paragraphs = paragraphs.lower()
            # title = title
            # _, title_list = self.stan_word_tokenize(nlp, title)
            # _, para_list = self.stan_word_tokenize(nlp, paragraphs)
            # if para_list == []:
            #     print('-------------paragraph---------')
            #     continue
            # input = [title_token] + title_list + [content_token] + para_list
            input = title_token + ' ' + title + ' ' + content_token + ' ' + paragraphs
            #print('Input:', input)
            #print('Abstract:', summary)

            # input, _ = self.find_author(input, norm_bl, author_token)
            # summary, substitute = self.find_author(summary, norm_bl, author_token)
            # print('input: ', input)
            # print('summary: ', summary)
            # print('byline:', norm_bl)
            # if truncated == True:
            #     if len(input) > max_len_in:
            #         input = input[:max_len_in]
            #     if len(summary) > max_len_sum:
            #         summary = summary[:max_len_sum]
            #input.append('EOS')
            #summary.append('EOS')
            input_list.append(input)
            output_list.append(summary)
            #tag_list.append(self.tag_onehot(tags))
            input_length += len(input)
            output_length += len(summary)
            new_news_dict.setdefault('input', input)
            new_news_dict.setdefault('summary', summary)
            new_news_dict.setdefault('topic', tags)
            new_all_dict.setdefault(key, new_news_dict)



        print(len(new_all_dict))
        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        return new_all_dict


    @staticmethod
    def zipall(input_list, output_list, tag_list):
        data_dict = {}
        id = 0
        for i, j, k in zip(input_list, output_list, tag_list):
            data_dict.setdefault(id, {
                'Input': i,
                'Summary': j,
                'Topic': k
            })
            id += 1
        return data_dict


    def train_withtopic_all(self, max_len_in = 400-1, max_len_sum = 100-1, intokennum=50000, outtokennum=50000,
                            write_file=False, numoftrain=10, filename='train_nyt.pkl',
                            makedict = True, truncated=True):
        nlp = StanfordCoreNLP('http://localhost:9000')
        print('host started')
        SOS_token = 0
        EOS_token = 1
        ban_list = ['photo','photos','graph','graphs','chart','charts','map','maps','table','tables','drawing','drawings']
        title_token = '</title>'
        byline_token = '</byline>'
        content_token = '</content>'
        author_token = '__Author__'
        #tokenizer = StanfordTokenizer()
        with open(filename,'rb') as f:
            train_dict = pickle.load(f)
        input_list = []
        output_list = []
        tag_list = []
        input_length = 0
        output_length = 0
        ly_i = 0
        word_count = {}
        for key, news_dict in train_dict.items():
            abstract = ''.join(news_dict['abstract']).lower()
            abstract = re.sub('\(m\)|\(s\)','',abstract)
            pattern = ';.?photo.?;|;.?graph.?;|;.?chart.?;|;.?map.?;|;.?table.?;|;.?drawing.?;'
            abstract = re.sub(pattern,'',abstract)
            abstract = re.sub('[0-9]{1,5}', '0', abstract)
            tags = news_dict['tag']

            headline = news_dict['hdl']
            online_sction = news_dict['online_section']
            date = news_dict['date']
            byline = news_dict['bl']
            nbl = news_dict['normbl']
            gd = news_dict['general_descriptors']
            descriptors = news_dict['descriptors']
            onhdl = news_dict['onhdl']



            _, summary = self.stan_word_tokenize(nlp, abstract)
            if summary == []:
                print('-------------summary---------')
                continue

            #_, summary_raw = self.stan_word_tokenize(nlp, abstract)
            if len(summary) > 5:
                if summary[-1] in ban_list and summary[-2] == ';':
                    summary.pop()
                    summary.pop()
            if len(summary) > 10:
                for i in range(5):
                    if summary[-i-1] in ban_list and summary[-i-2] == ';':
                        for j in range(i+2):
                            summary.pop()
                        break
            title = ''.join(news_dict['hdl'])
            byline = ''.join(news_dict['bl']).lower()
            norm_bl = ''.join(news_dict['normbl']).lower()
            if byline == None:
                byline = ''
            if norm_bl == None:
                norm_bl = ''

            _, norm_bl = self.stan_word_tokenize(nlp, ''.join(norm_bl.split(',')))
            _, bl_token = self.stan_word_tokenize(nlp, byline)

            for i in range(len(bl_token)):
                if bl_token[i] == 'by':
                    try:
                        norm_bl.append(bl_token[i+1])
                    except IndexError:
                        continue
                    try:
                        norm_bl.append(bl_token[i+2])
                    except IndexError:
                        continue
                    try:
                        if bl_token[i+3] != '(':
                            norm_bl.append(bl_token[i+3])
                    except IndexError:
                        continue


            assert title != None
            paragraphs = '\n'.join(news_dict['paragraphs'])
            paragraphs = re.sub('\'\'','',paragraphs)
            paragraphs = re.sub('[0-9]{1,5}', '0', paragraphs)
            title = re.sub('[0-9]{1,5}', '0', title)
            title = re.sub('\'\'','',title)
            paragraphs = paragraphs.lower()
            title = title.lower()
            _, title_list = self.stan_word_tokenize(nlp, title)
            _, para_list = self.stan_word_tokenize(nlp, paragraphs)
            if para_list == []:
                print('-------------paragraph---------')
                continue
            input = [title_token] + title_list + [content_token] + para_list

            input, _ = self.find_author(input, norm_bl, author_token)
            summary, substitute = self.find_author(summary, norm_bl, author_token)
            # print('input: ', input)
            # print('summary: ', summary)
            # print('byline:', norm_bl)
            if len(input) > max_len_in:
                input = input[:max_len_in]
            if truncated == True:
                if len(summary) > max_len_sum:
                    summary = summary[:max_len_sum]
            #input.append('EOS')
            #summary.append('EOS')
            input_list.append(input)
            output_list.append(summary)
            tag_list.append(self.tag_onehot(tags))
            input_length += len(input)
            output_length += len(summary)
            for word in input + summary:
                try:
                    word_count[word] += 1
                except KeyError:
                    word_count.setdefault(word, 1)
            ly_i += 1
            if ly_i > numoftrain - 1:
                break




        print('train_number: ', len(input_list))
        print('train_number: ', len(output_list))
        print('average input length: ', input_length/len(input_list))
        print('average summary length: ', output_length/len(output_list))
        def make_dict(word_count, tokennum):
            if tokennum < len(word_count):
                input_word = sorted(word_count.items(),key=lambda x:x[1],reverse=True)[:tokennum]
            else:
                input_word = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            in_tkset = [token for token,_ in input_word]
            in_tkset.insert(0, 'UNK')
            in_tkset.insert(0, 'EOS')
            in_tkset.insert(0, 'SOS')
            in_tkset.insert(0, 'PAD')
            in_tk2id = {in_tkset[i]:i for i in range(len(in_tkset))}
            in_id2tk = {i:in_tkset[i] for i in range(len(in_tkset))}
            return in_tk2id, in_id2tk
        if makedict:
            in_tk2id, in_id2tk = make_dict(word_count, intokennum)
            out_tk2id, out_id2tk = make_dict(word_count, outtokennum)

            print('Number of intokens: ', len(in_tk2id))
            print('Number of outtokens: ', len(out_tk2id))
            print('Number of intokens: ', len(in_id2tk))
            print('Number of outtokens: ', len(out_id2tk))
            return in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list, tag_list
        else:
            return input_list, output_list, tag_list





if __name__ == '__main__':
    preprocess = Preprocessing()
    #preprocess.filter_zero_tag()
    #preprocess.train_test_split()
    #in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list = preprocess.filter()
    #tokenizer = StanfordTokenizer()
    #in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list = preprocess.train_preprocess_mini()
    dir_name = '/data/'
    # train_new = preprocess.make_all_dict(numoftrain=1000000, filename='/data/luyang/nyt_data_filter0abstract.pkl',
    #                                                                                                              truncated=False,
    #                                                                                                              makedict=True)

    train_new = preprocess.make_all_dict_noparsing(numoftrain=1000000, filename='/data/luyang/nyt_data_filter0abstract.pkl',
                                                                                                                 truncated=False,
                                                                                                                 makedict=True)

    # train_file = open(dir_name+'train_processed_topic.pkl','wb')
    # pickle.dump(train_all, train_file)
    # train_file.close()
    train_file_2 = open(dir_name+'luyang/nyt-wo-author.pkl', 'wb')

    pickle.dump(train_new, train_file_2)
    train_file_2.close()


    # filename = 'train_processed_topic.pkl'
    # train_file = open(filename, 'rb')
    # train_all = pickle.load(train_file)
    # in_tk2id, in_id2tk, out_tk2id, out_id2tk, _, _, _ = train_all
    # train_file.close()
    #
    # input_list, output_list, tag_list = preprocess.train_withtopic_all(numoftrain=600000, filename='test_nyt.pkl', truncated=False, makedict=False)
    # count = 0
    # for output in output_list:
    #     if len(output) > 100:
    #         count += 1
    # print('Count:', count)
    # train_all = [in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list, tag_list]
    # train_file = open(dir_name+'test_processed_topic.pkl','wb')
    # pickle.dump(train_all, train_file)
    # train_file.close()
    # train_file_2 = open(dir_name+'test_dict.pkl', 'wb')
    # train_new = preprocess.zipall(input_list, output_list, tag_list)
    # pickle.dump(train_new, train_file_2)
    # train_file_2.close()
    #
    # input_list, output_list, tag_list = preprocess.train_withtopic_all(numoftrain=600000, filename='dvp_nyt.pkl')
    # train_all = [in_tk2id, in_id2tk, out_tk2id, out_id2tk, input_list, output_list, tag_list]
    # train_file = open(dir_name+'dvp_processed_topic.pkl','wb')
    # pickle.dump(train_all, train_file)
    # train_file.close()
    # train_file_2 = open(dir_name+'dvp_dict.pkl', 'wb')
    # train_new = preprocess.zipall(input_list, output_list, tag_list)
    # pickle.dump(train_new, train_file_2)
    # train_file_2.close()