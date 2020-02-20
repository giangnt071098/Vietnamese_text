import os
from tqdm import tqdm
import re
PATH_DATA = "./output"

alphabet = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'

list_sub_folder = os.listdir(PATH_DATA)

for sub_folder in (list_sub_folder):
    path_sub_folder = os.path.join(PATH_DATA, sub_folder)
    list_file = os.listdir(path_sub_folder)
    for file in tqdm(list_file):
        with open(os.path.join(path_sub_folder, file), "r",encoding='utf-8') as f_r:
            contents = f_r.read()
            contents = re.sub("(\s)+", r"\1", contents)
            contents = contents.split("\n")
            for content in contents:
                try:
                    content = eval(content)
                except:
                    continue
                lines = content["text"].split("\n")
                with open("./train_data2.txt", "a",encoding='utf-8') as f_w:
                    for line in lines[1:]:
                        if len(line.split()) > 2 and re.match(alphabet, line.lower()):
                            f_w.write(line + "\n")