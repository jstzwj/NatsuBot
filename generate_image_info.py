import glob
import os
import json


if __name__ == "__main__":
    img_list = glob.glob('./images/*.gif')
    for each_img in img_list:
        each_path = './images/' + os.path.basename(each_img) + '.json'
        if not os.path.exists(each_path):
            with open(each_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'labels':[]}))