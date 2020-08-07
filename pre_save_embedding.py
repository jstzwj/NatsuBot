import pickle
import os
import json
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer('distiluse-base-multilingual-cased', device='cuda')
    # path = ['./chat_text/poetry.txt', './chat_text/chat_text.txt']
    # path = ['./chat_text/chat_text.txt', './chat_text/ownthink_v2.txt']
    path = ['./chat_text/basic_settings.jsonl', './chat_text/natsu_chat.jsonl']
    limit = 300000
    count = 0
    if os.path.isfile('embeddings.pickle'):
        with open('embeddings.pickle', 'rb') as file:
            embedding_cache = pickle.load(file)
    else:
        embedding_cache = {}
    for each_path in path:
        print(each_path)
        with open(each_path, 'r', encoding='utf-8') as f:
            questions = []
            for each_line in f:
                obj = json.loads(each_line)
                question, answer = obj['question'], obj['answer']
                if 'context' in obj:
                    context = obj['context']
                    for each_context in context:
                        if each_context not in embedding_cache:
                            questions.append(each_context)
                if question not in embedding_cache:
                    questions.append(question)

                if len(questions) > 4096 * 10:
                    out = model.encode(questions, batch_size=512, show_progress_bar=True)
                    for i, each_out in enumerate(out):
                        embedding_cache[questions[i]] = each_out
                    questions.clear()
                count += 1
                if count > limit:
                    break
            # 把剩余question全部处理完
            if len(questions) != 0:
                out = model.encode(questions, show_progress_bar=True)
                for i, each_out in enumerate(out):
                    embedding_cache[questions[i]] = each_out
                questions.clear()
            if count > limit:
                    break
    with open('embeddings.pickle', 'wb') as file:
        pickle.dump(embedding_cache, file)