import os
from pyrouge import Rouge155
from nltk import sent_tokenize
import glob

my_dir = 'testout/my_sum_s2s/'
my_dir2 = 'testout/my_sum_rl/'
# my_dir3 = 'testout/my_sum_mlp_setup2/'

gold_dir = 'testout/gold/'
if not os.path.exists(my_dir):
    os.mkdir(my_dir)
else:
    files = glob.glob(my_dir + '*.txt')
    for file in files:
        os.remove(file)
if not os.path.exists(my_dir2):
    os.mkdir(my_dir2)
else:
    files = glob.glob(my_dir2 + '*.txt')
    for file in files:
        os.remove(file)
# if not os.path.exists(my_dir3):
#     os.mkdir(my_dir3)
# else:
#     files = glob.glob(my_dir3 + '*.txt')
#     for file in files:
#         os.remove(file)
if not os.path.exists(gold_dir):
    os.mkdir(gold_dir)
else:
    files = glob.glob(gold_dir + '*.txt')
    for file in files:
        os.remove(file)

test_file1 = 'testout/test-bl.txt'
test_file2 = 'testout/test-rl.txt'
# test_file3 = 'testout/test-mlp-dec-setup2.txt'
gold_file = 'testout/gold.txt'
print(my_dir)

print(gold_dir)
print(test_file1)

print(gold_file)

# if not os.path.exists('rouge/gold_sum'):
#     os.mkdir('rouge/gold_sum')
# if not os.path.exists('rouge/my_sum'):
#     os.mkdir('rouge/my_sum')
error = 0
count = 0
with open(test_file1, 'r', encoding='utf-8') as f:
    id = 1
    for line in f:
        count += 1
        sents = line.strip().replace("</title>","").replace("<title>","").split('</t>')
        try:
            sents.remove('')
        except:
            sents = sents
        summary = '\n'.join(sents)
        if len(summary) != 0:
            f1 = open(my_dir + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
            f1.write('\n'.join(sents))
            f1.close()
            id += 1
        else:
            f1 = open(my_dir + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
            f1.write('' + '\n')
            f1.close()
            id += 1
            error += 1

print('Error Num: ', error)
print('total: ', count)

error = 0
count = 0
with open(test_file2, 'r', encoding='utf-8') as f:
    id = 1
    for line in f:
        count += 1
        sents = line.strip().replace("</title>","").replace("<title>","").split('</t>')
        try:
            sents.remove('')
        except:
            sents = sents
        summary = '\n'.join(sents)
        if len(summary) != 0:
            f1 = open(my_dir2 + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
            f1.write('\n'.join(sents))
            f1.close()
            id += 1
        else:
            f1 = open(my_dir2 + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
            f1.write('' + '\n')
            f1.close()
            id += 1
            error += 1

print('Error Num: ', error)
print('total: ', count)

error = 0
with open(gold_file, 'r', encoding='utf-8') as f:
    id = 1
    for line in f:
        sents = line.strip().replace("*","").replace('�', '').split('</t>')
        try:
            sents.remove('')
        except:
            sents = sents
        if len(sents) == 0:
            sents = '__Author__'
        summary = '\n'.join(sents)
        if len(summary) != 0:
            f1 = open(gold_dir + 'gold.A.%05d.txt' % id, 'w', encoding='utf-8')
            f1.write('\n'.join(sents))
            f1.close()
            id += 1
        else:
            f1 = open(gold_dir + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
            f1.write('' + '\n')
            f1.close()
            id += 1
            error += 1

print('Error Num: ', error)

# error = 0
# count = 0
# with open(test_file3, 'r', encoding='utf-8') as f:
#     id = 1
#     for line in f:
#         count += 1
#         sents = line.strip().replace("</title>","").replace("</content>","").split('</t>')
#         try:
#             sents.remove('')
#         except:
#             sents = sents
#         summary = '\n'.join(sents)
#         if len(summary) != 0:
#             f1 = open(my_dir3 + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
#             f1.write('\n'.join(sents))
#             f1.close()
#             id += 1
#         else:
#             f1 = open(my_dir3 + 'pred.%05d.txt' % id, 'w', encoding='utf-8')
#             f1.write('' + '\n')
#             f1.close()
#             id += 1
#             error += 1
#
# print('Error Num: ', error)
# print('total: ', count)


# with open('rouge/test-tgt.txt', 'r', encoding='utf-8') as f:
#     id = 1
#     for line in f:
#         f1 = open('rouge/gold_sum/gold.A.%05d.txt'%id, 'w', encoding='utf-8')
#         f1.write('\n'.join(sent_tokenize(line)))
#         f1.close()
#         id += 1


# config_file_path = 'pyrouge/rouge_config.xml'
# Rouge155.write_config_static(
#     pred_dir, 'pred.[A-Z].(\d+).txt',
#     gold_dir, 'gold.(\d+).txt',
#     config_file_path)

pred_dir = my_dir
r = Rouge155()
r.system_dir = pred_dir
r.model_dir = gold_dir
r.system_filename_pattern = 'pred.(\d+).txt'
r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'

output = r.convert_and_evaluate(system_id=0)
print('seq2seq  : ')
print(output)
output_dict = r.output_to_dict(output)


pred_dir = my_dir2
r = Rouge155()
r.system_dir = pred_dir
r.model_dir = gold_dir
r.system_filename_pattern = 'pred.(\d+).txt'
r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print('pg: ')
print(output)
output_dict = r.output_to_dict(output)

# pred_dir = my_dir3
# r = Rouge155()
# r.system_dir = pred_dir
# r.model_dir = gold_dir
# r.system_filename_pattern = 'pred.(\d+).txt'
# r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'
#
# output = r.convert_and_evaluate()
# print('MLP: ')
# print(output)
# output_dict = r.output_to_dict(output)
