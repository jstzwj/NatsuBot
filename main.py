from mirai import Mirai, MessageChain,\
    Friend, Member, Group, Source, MemberMuteEvent, BotJoinGroupEvent, \
    Plain, At, AtAll, Image, Face, XmlMessage, JsonMessage
import asyncio
import json
import os
import re
import random
import datetime
import uuid
import glob
import tqdm
import numpy as np
import pickle
import requests
import bs4
from typing import Dict, List, Iterable
from model import *
import sklearn
from sklearn.neighbors import KDTree
from mingan import mingan_replace

import jieba
from sentence_transformers import SentenceTransformer

with open('settings.json', 'r', encoding='utf-8') as f:
    settings = json.loads(f.read())

qq = settings['qq'] # 字段 qq 的值
authKey = settings['authKey'] # 字段 authKey 的值
mirai_api_http_locate = settings['mirai_api_http_locate'] # httpapi所在主机的地址端口,如果 setting.yml 文件里字段 "enableWebsocket" 的值为 "true" 则需要将 "/" 换成 "/ws", 否则将接收不到消息.

app = Mirai(f"mirai://{mirai_api_http_locate}?authKey={authKey}&qq={qq}")

# 主动发送消息的群
active_group = settings['active_group']


class QQMessage(object):
    def __init__(self):
        self.group = None
        self.member = None
        self.friend = None
        self.msg_type = None
        self.time = None
        self.message = []

    @classmethod
    def from_message_chain(cls, message:MessageChain, msg_type:str, group=None, member=None, friend=None):
        ret = QQMessage()
        ret.member = member
        ret.friend = friend
        ret.msg_type = msg_type
        ret.group = group
        ret.time = datetime.datetime
        for nc in message.__root__:
            if nc == message.__root__[0]:
                continue
            json_msg = None
            if isinstance(nc, Plain):
                json_msg = {
                    'type': 'Plain',
                    'text': nc.text
                }
            elif isinstance(nc, At):
                json_msg = {
                    'type': 'At',
                    'target': nc.target,
                    'display': nc.display
                }
            elif isinstance(nc, AtAll):
                json_msg = {
                    'type': 'AtAll',
                }
            elif isinstance(nc, Image):
                json_msg = {
                    'type': 'Image',
                    'imageId': nc.imageId,
                    'url': str(nc.url)
                }
            elif isinstance(nc, Face):
                json_msg = {
                    'type': 'Face',
                    'faceId': nc.faceId,
                    'name': nc.name
                }
            elif isinstance(nc, XmlMessage):
                json_msg = {
                    'type': 'Xml',
                    'xml': nc.XML,
                }
            elif isinstance(nc, JsonMessage):
                json_msg = {
                    'type': 'Json',
                    'json': nc.Json,
                }
            ret.message.append(json_msg)
        return ret

    def to_message_list(self):
        ret = []
        for each_component in self.message:
            if each_component['type'] == 'Plain':
                ret.append(Plain(text=each_component['text']))
            elif each_component['type'] == 'At':
                ret.append(At(target=each_component['target']))
            elif each_component['type'] == 'AtAll':
                ret.append(AtAll())
            elif each_component['type'] == 'Image':
                ret.append(Image(imageId=each_component['imageId']))
            elif each_component['type'] == 'Face':
                ret.append(Face(faceId=each_component['faceId']))
            elif each_component['type'] == 'Xml':
                ret.append(Xml(XML=each_component['xml']))
            elif each_component['type'] == 'Json':
                ret.append(Json(Json=each_component['json']))

        return ret

    def toString(self):
        ret = ''
        for each_component in self.message:
            if each_component is None:
                continue
            if each_component['type'] == 'Plain':
                ret += each_component['text']

        return ret

    def get_first_component(self, cls_name:str):
        ret = None
        for each_component in self.message:
            if each_component is None:
                continue
            if each_component['type'] == cls_name:
                ret = each_component
                break
        
        return ret

class BaseChatComponent(object):
    def __init__(self):
        pass

    async def onReceive(self, app: Mirai, kind: str, qq_message:QQMessage):
        pass


class SimpleRepeatChatComponent(BaseChatComponent):
    def __init__(self):
        pass

    async def onReceive(self, app: Mirai, kind: str, qq_message:QQMessage):
        await asyncio.sleep(random.randint(1, 5))
        if kind=='t':
            await app.sendTempMessage(qq_message.group, qq_message.member, qq_message.to_message_list())
        elif kind == 'f':
            await app.sendFriendMessage(qq_message.friend, qq_message.to_message_list())
        elif kind == 'g':
            await app.sendFriendMessage(qq_message.group, qq_message.member, qq_message.to_message_list())


class HistoryComponent(BaseChatComponent):
    def __init__(self):
        pass

    async def onReceive(self, app: Mirai, kind: str, qq_message:QQMessage):
        if kind=='t':
            record = Chat.create(user=qq_message.member.id, group=qq_message.group.id, kind=kind, message=json.dumps(qq_message.message))
            record.save()
        elif kind == 'f':
            record = Chat.create(user=qq_message.friend.id, kind=kind, message=json.dumps(qq_message.message))
            record.save()
        elif kind == 'g':
            record = Chat.create(user=qq_message.member.id, group=qq_message.group.id, kind=kind, message=json.dumps(qq_message.message))
            record.save()

class RepeatChatComponent(BaseChatComponent):
    def __init__(self):
        self.history_message = {}

    async def onReceive(self, app: Mirai, kind: str, qq_message:QQMessage):
        if kind != 'g':
            return
        await asyncio.sleep(random.randint(1, 5))

        group_id = qq_message.group.id
        if group_id not in self.history_message:
            self.history_message[group_id] = []
        
        group_queue = self.history_message[group_id]
        group_queue.append(qq_message)

        if len(group_queue) >= 3 and \
            group_queue[-1].message == group_queue[-2].message and \
            group_queue[-2].message == group_queue[-3].message:
            await app.sendGroupMessage(qq_message.group, qq_message.to_message_list())
            group_queue.clear()

        if len(group_queue) > 4:
            group_queue.pop(0)


class TrainComponent(BaseChatComponent):
    def __init__(self):
        self.history_message = {}
        self.train_id = [691310634, 794424922]

    async def onReceive(self, app: Mirai, kind: str, qq_message:QQMessage):
        if kind != 'g':
            return
        await asyncio.sleep(random.randint(1, 5))

        group_id = qq_message.group.id
        if group_id not in self.history_message:
            self.history_message[group_id] = []
        
        group_queue = self.history_message[group_id]
        group_queue.append(qq_message)

        if len(group_queue) >= 3 and \
            group_queue[-1].message == group_queue[-2].message and \
            group_queue[-2].message == group_queue[-3].message:
            await app.sendGroupMessage(qq_message.group, qq_message.to_message_list())
            group_queue.clear()

        if len(group_queue) > 4:
            group_queue.pop(0)

class GeneralChatComponent(BaseChatComponent):
    def __init__(self):
        self.name = settings['bot_name']
        self.short_name = settings['bot_short_name']
        self.path = ['./chat_text/basic_settings.jsonl', './chat_text/natsu_chat.jsonl']
        self.id = settings['bot_id']
        
        self.model = SentenceTransformer('distiluse-base-multilingual-cased')

        self.qa = {}
        self.questions = []
        self.embedding_cache = []
        self.load_embedding_cache()
        self.kdtree = KDTree(np.array(self.embedding_cache), leaf_size=40)

        self.label2image = {}
        self.load_images_cache()

        self.context = []
        self.context_window = 3

    def load_images_cache(self):
        img_info_list = glob.glob('./images/*.json')
        for each_info in img_info_list:
            with open(each_info, 'r', encoding='utf-8') as f:
                obj = json.loads(f.read())
            labels = obj['labels']
            for each_label in labels:
                if each_label not in self.label2image:
                    self.label2image[each_label] = []
                self.label2image[each_label].append(each_info[:-5])


    def load_embedding_cache(self, limit=20000):
        if os.path.exists('embeddings.pickle'):
            with open('embeddings.pickle', 'rb') as file:
                cache = pickle.load(file)
        else:
            cache = {}
        print(f'加载得{len(cache)}条问答对')
        count = 0
        for each_path in self.path:
            with open(each_path, 'r', encoding='utf-8') as f:
                questions = []
                for each_line in f:
                    obj = json.loads(each_line)
                    question = obj['question']
                    self.qa[question] = obj
                    if question in cache:
                        self.questions.append(question)
                        self.embedding_cache.append(cache[question])
                    else:
                        questions.append(question)

                    if len(questions) > 64:
                        out = self.model.encode(questions, show_progress_bar=True)
                        for i, each_out in enumerate(out):
                            self.questions.append(questions[i])
                            self.embedding_cache.append(each_out)
                        questions.clear()
                    count += 1
                    if count > limit:
                        return
                if len(questions) != 0:
                    out = self.model.encode(questions, show_progress_bar=True)
                    for i, each_out in enumerate(out):
                        self.questions.append(questions[i])
                        self.embedding_cache.append(each_out)
                    questions.clear()

    def get_features(self, text):
        words = jieba.cut(text)
        tokens = [each_token for each_token in words]
        return tokens

    async def send_message(self, app: Mirai, kind: str, qq_message:QQMessage, msg:str):
        msg = mingan_replace(msg)
        if kind == 'g':
            await app.sendGroupMessage(qq_message.group, msg)
        elif kind == 'f':
            await app.sendFriendMessage(qq_message.friend, msg)
        elif kind == 't':
            await app.sendTempMessage(qq_message.group, qq_message.member, msg)
    
    async def send_pic(self, app: Mirai, kind: str, qq_message:QQMessage, pic_path:str):
        print('send pic: ' + pic_path)
        if kind == 'g':
            await app.sendGroupMessage(qq_message.group, [Image.fromFileSystem(pic_path)])
        elif kind == 'f':
            await app.sendFriendMessage(qq_message.friend, [Image.fromFileSystem(pic_path)])
        elif kind == 't':
            await app.sendTempMessage(qq_message.group, qq_message.member, [Image.fromFileSystem(pic_path)])

    async def baidu_search(self, app: Mirai, kind: str, qq_message:QQMessage, question:str):
        baidu_query_url = 'https://www.baidu.com/s?wd=' + question + '&usm=3&rsv_idx=2&rsv_page=1'
        answer = ''
        try:
            headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
            r = requests.get(baidu_query_url, headers=headers)
            r.encoding = 'utf-8'
            html = bs4.BeautifulSoup(r.text, features="lxml")

            result_list = html.find_all(name='div',attrs={"class":"c-abstract"})
            if len(result_list) <= 5:
                return answer
            first_result = result_list[4]
            first_result_text = first_result.text

            first_idx = 0
            for i in range(len(first_result_text)):
                if first_result_text[i] in ['.', '。']:
                    first_idx = i
                    break
            answer = first_result_text[:first_idx]
            # specal case
            if answer.startswith('最佳答案:'):
                answer = answer[5:]
            if answer.startswith('20'):
                answer = answer[13:]

            if len(answer) > 50:
                answer = answer[:50]
        except requests.exceptions.ConnectionError as e:
            print(e)

        return answer

    async def random_chat(self, app: Mirai, kind: str, qq_message:QQMessage, always_chat=True):
        question = qq_message.toString()
        if question.startswith(f'@{self.short_name}'):
            question = question[len(self.short_name) + 1:].strip()

        if question.startswith(f'{self.short_name}，'):
            question = question[len(self.short_name) + 1:]
        elif question.startswith(f'{self.name}，'):
            question = question[len(self.name) + 1:]
        elif question.startswith(f'{self.short_name}'):
            question = question[len(self.short_name):].strip()
        elif question.startswith(f'{self.name}'):
            question = question[len(self.name):].strip()

        # 正则表达式匹配url
        pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        if re.match(pattern, question) is not None:
            try:
                headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36"}
                r = requests.get(question, headers=headers)
                html = bs4.BeautifulSoup(r.text, features="lxml")
                question = html.title.text
            except requests.exceptions.ConnectionError as e:
                print(e)

        # 获得发图命令
        # print(question)
        if random.randint(0, 1) == 0:
            if '萌' in question or '图' in question or '表情' in question:
                houxuan_imgs = []
                for each_label, each_path in self.label2image.items():
                    houxuan_imgs.extend(each_path)
                random.shuffle(houxuan_imgs)
                await self.send_pic(app, kind, qq_message, houxuan_imgs[0])
                return

        # 空数据直接返回
        if len(question) == 0:
            return

        # 添加到历史纪录
        self.context.append(question)
        while len(self.context) >= 50:
            self.context.pop(0)

        # 直接搜索
        if question in self.qa:
            obj = self.qa[question]
            answer = obj['answer']
            if 'script' in obj:
                local_dict = {'obj': obj, 'answer': answer, 'qq_message':qq_message}
                exec(obj['script'], globals(), local_dict)
                answer = local_dict['answer']

            if always_chat:
                # self.context.append(answer)
                await app.sendGroupMessage(qq_message.group, answer)
            else:
                if random.randint(0, 5) == 0:
                    # self.context.append(answer)
                    await app.sendGroupMessage(qq_message.group, answer)
            return

        # 近似搜索
        # 语义
        question_embedding = np.array(self.model.encode([question])[0])
        context_embeddings = self.model.encode(self.context)
        context_embedding = np.zeros((512,))
        embed_count = 0
        for i in reversed(range(len(self.context))):
            order = len(self.context) - i
            if order > self.context_window:
                break
            context_embedding += np.array(context_embeddings[i])
            embed_count += 1
        main_embedding = (question_embedding + context_embedding / embed_count) / 2

        dist, ind = self.kdtree.query(np.array(main_embedding).reshape(1, -1), k=3)
        dist = dist[0,:]
        ind = ind[0,:]
        
        if len(self.context) < 30:
            for each_pred in range(len(dist)):
                '''
                if self.qa[self.questions[ind[each_pred]]]['answer'] in self.context:
                    continue
                '''
                selected_result = each_pred
                break
        else:
            for each_pred in range(len(dist)):
                '''
                if self.qa[self.questions[ind[each_pred]]]['answer'] in self.context[-30:]:
                    continue
                '''
                selected_result = each_pred
                break

        min_dist = dist[selected_result]
        min_question = ind[selected_result]

        print([(self.qa[self.questions[ind[i]]], dist[i]) for i in range(len(dist))])

        obj = self.qa[self.questions[min_question]]
        answer = obj['answer']
        if 'script' in obj:
            local_dict = {'obj': obj, 'answer': answer, 'qq_message':qq_message}
            exec(obj['script'], globals(), local_dict)
            answer = local_dict['answer']

        if not always_chat:
            if min_dist < 0.5 and random.randint(0, 3) == 0:
                # self.context.append(answer)
                await self.send_message(app, kind, qq_message, answer)
            elif min_dist < 0.7 and random.randint(0, 2) == 0:
                houxuan_imgs = []
                for each_label, each_path in self.label2image.items():
                    if each_label in answer:
                        houxuan_imgs.extend(each_path)
                if len(houxuan_imgs) > 0:
                    random.shuffle(houxuan_imgs)
                    await self.send_pic(app, kind, qq_message, houxuan_imgs[0])
            '''
            elif random.randint(0, 4) == 0:
                answer = await self.baidu_search(app, kind, qq_message, question)
                await app.sendGroupMessage(qq_message.group, answer)
            '''
        else:
            if random.randint(0, 3) == 0:
                houxuan_imgs = []
                for each_label, each_path in self.label2image.items():
                    if each_label in answer:
                        houxuan_imgs.extend(each_path)
                if len(houxuan_imgs) > 0:
                    random.shuffle(houxuan_imgs)
                    await self.send_pic(app, kind, qq_message, houxuan_imgs[0])
            else:
                # self.context.append(answer)
                await self.send_message(app, kind, qq_message, answer)

    async def onReceive(self, app: Mirai, kind: str, qq_message:QQMessage):
        # if kind != 'g':
        #    return

        question = qq_message.toString().lstrip()

        first_at = qq_message.get_first_component('At')
        if first_at is None or first_at['target'] != self.id:
            if question.startswith('@kkr'):
                question = question[4:].strip()
                always_chat = True
            elif question.startswith('kkr'):
                question = question[3:].strip()
                always_chat = True
            elif question.startswith('可可萝'):
                question = question[3:].strip()
                always_chat = True
            else:
                always_chat = False
        else:
            always_chat = True

        await asyncio.sleep(random.randint(1, 5))

        await self.random_chat(app, kind, qq_message, always_chat)

components = [
    RepeatChatComponent(),
    HistoryComponent(),
    GeneralChatComponent(),
]

@app.receiver("TempMessage")
async def event_temp(app: Mirai, message: MessageChain, group: Group, member: Member):
    '''
    await app.sendFriendMessage(friend, [
        Plain(text="Hello, world!")
    ])
    '''
    qq_message = QQMessage.from_message_chain(message, msg_type='t', group=group, member=member)

    for each_component in components:
        await each_component.onReceive(app, kind='t', qq_message=qq_message)

@app.receiver("FriendMessage")
async def event_fm(app: Mirai, message: MessageChain, friend: Friend):
    '''
    await app.sendFriendMessage(friend, [
        Plain(text="Hello, world!")
    ])
    '''
    qq_message = QQMessage.from_message_chain(message, msg_type='f', friend=friend)

    for each_component in components:
        await each_component.onReceive(app, kind='f', qq_message=qq_message)

@app.receiver("GroupMessage")
async def group_recv(app: Mirai, message: MessageChain, group: Group, member: Member, source: Source):
    qq_message = QQMessage.from_message_chain(message, msg_type='g', group=group, member=member)

    for each_component in components:
        await each_component.onReceive(app, kind='g', qq_message=qq_message)


@app.onStage("start")
async def start_stage_subroutine(app: Mirai):
    pass
    '''
    for each_group in active_group:
        await app.sendGroupMessage(each_group, '我回来啦')
    '''

@app.onStage("end")
async def end_stage_subroutine(app: Mirai):
    pass
    '''
    for each_group in active_group:
        await app.sendGroupMessage(each_group, '休息了，一会儿见喔')
    '''

@app.subroutine
async def subroutine_timer(app: Mirai):
    await asyncio.sleep(3600)
    dt = datetime.datetime.now()
    if dt.hour > 1 and dt.hour < 5:
        for each_group in active_group:
            await app.sendGroupMessage(each_group, f'凌晨{dt.hour}点了呢，大家早点睡喔') 

if __name__ == "__main__":
    app.run()