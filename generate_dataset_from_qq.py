
import re
import tqdm
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distiluse-base-multilingual-cased', device='cpu')

re_charset = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9]")
re_message = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) (.*)(\(.+\)|<.+>)')

def read_qq_history_file(filename):
    message = []
    # get line count
    count = -1
    for count, line in enumerate(open(filename, 'r', encoding='utf-8')):
        pass
    count += 1

    # read data
    with open(filename, 'r', encoding='utf-8') as f:
        for i in range(8):
            header = f.readline()
            print(header)

        cur_msg = None

        line = f.readline()
        with tqdm.tqdm(total=count, ascii=True) as pbar:
            while line:
                if line.strip() == '':
                    line = f.readline()
                    continue
                msg_header = re_message.match(line.strip())
                if msg_header:
                    if cur_msg is not None:
                        message.append(cur_msg)

                    cur_msg = {
                        'time': msg_header.group(1),
                        'user': msg_header.group(2),
                        'id': msg_header.group(3),
                        'data': ''
                    }
                else:
                    cur_msg['data'] += line
                line = f.readline()
                pbar.update()
    
    return message


def filter_msg(messages):
    ret = []
    with tqdm.tqdm(total=len(messages), ascii=True) as pbar:
        last_message = None
        for each_msg in messages:
            data = each_msg['data']
            data = data.replace("\n", '')
            data = data.replace(r'[图片]', '')
            data = data.replace(r'[表情]', '')
            data = re.sub(r'(http|https|ftp)://[0-9a-zA-Z~./_\-]+', '', data)
            data = re.sub(r'@.+ ', '', data)
            data = re.sub(r'.+加入本群', '', data)
            data = re.sub(r'.+被管理员禁言[0-9]{1,2}(分钟|天)', '', data)
            data = re.sub(r'.+被管理员解除禁言', '', data)
            data = re.sub(r'.+撤回了一条消息', '', data)
            data = re.sub(r'\[礼物\] .+成为.+的守护者', '', data)
            data = re.sub(r'\[送礼物\] 为.+', '', data)
            data = re.sub(r'\[QQ红包\]我发了一个.*', '', data)
            data = re.sub(r'\[动作消息\].+', '', data)
            if last_message is not None and last_message['data'] == data:
                continue

            # data = re_charset.sub("", data)
            filtered_msg = {
                        'time': each_msg['time'],
                        'user': each_msg['user'],
                        'id': each_msg['id'],
                        'data': data
                    }
            if filtered_msg['data'].strip() != '':
                ret.append(filtered_msg)
                last_message = filtered_msg
            pbar.update()
    return ret


def generate_dataset(messages, output_path, person):
    context_size = 5
    with open(output_path, 'a', encoding='utf-8') as f:
        with tqdm.tqdm(total=len(messages), ascii=True) as pbar:
            for i, each_msg in enumerate(messages):
                if each_msg['data'].strip() == '':
                    continue
                if i < context_size:
                    continue
                # filter conditions
                if person is None or each_msg['id'] == person:
                    context = [each_data['data'] for each_data in messages[i-context_size:i]]
                    # 找到准确的question
                    context_embed = model.encode(context)
                    answer_embed = model.encode([each_msg['data']])[0]
                    dists = []
                    for each_context_embed in context_embed:
                        dist = np.linalg.norm(np.array(answer_embed) - np.array(each_context_embed))
                        dists.append(dist)
                    min_idx = np.argmin(np.array(dists))
                    # 输出
                    f.write(json.dumps({
                        'context': context,
                        'question': context[min_idx],
                        'answer': each_msg['data']
                    }, ensure_ascii=False) + '\n')

                pbar.update()

if __name__ == "__main__":
    print('read message')
    msg = read_qq_history_file('chat.txt')
    print('filter message')
    msg = filter_msg(msg)
    print('write to file')
    person = '<chino@hotococoa.moe>'
    # person = None
    generate_dataset(msg, 'chat_out.jsonl', person)
