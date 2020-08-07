'''
敏感词删除
'''

import re
import glob
dict_path = "./mingan_data/*.txt"
bad_words = set()
files = glob.glob(dict_path)
for each_file in files:
    with open(each_file, 'r', encoding='utf-8') as f:
        for each_line in f:
            word = each_line.strip()
            word = word.replace('(', '\\(')
            word = word.replace(')', '\\)')
            word = word.replace('[', '\\[')
            word = word.replace(']', '\\]')
            word = word.replace('{', '\\{')
            word = word.replace('}', '\\}')
            word = word.replace('.', '\\.')
            word = word.replace('+', '\\+')
            word = word.replace('*', '\\*')
            word = word.replace('?', '\\?')
            bad_words.add(word)

bad_words = list(bad_words)

re_string = ''
for i, each_word in enumerate(bad_words):
    if i == 0:
        re_string += each_word
    else:
        re_string += ('|' + each_word)
    
prog = re.compile(re_string)

def mingan_replace(words):
    ret = list(words)
    for m in prog.finditer(words):
        for i in range(m.start(),m.end()):
            ret[i] = '*'

    return "".join(ret)

if __name__ == "__main__":
    text = '中共中央2020年7月6日'
    print(mingan_replace(text))