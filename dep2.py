# conda activate D:\anaconda\envs\const
# C:\Users\ASUS\OneDrive\桌面\indo-dotGCN\dep2.py
from trankit import Pipeline
nlp = Pipeline('english')
import numpy as np
datasets = ["semeval14\\laptop_test.raw","semeval14\\laptop_train.raw",
"semeval16\\restaurant_test.raw", "semeval16\\restaurant_train.raw",
"T_data\\train.raw", "T_data\\test.raw"]

def get_dep(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    all_tag_path = []
    for i in range(0, len(lines), 3):
        left_and_right = lines[i].strip().split("$T$")
        sentence = left_and_right[0] +lines[i+1].strip() + left_and_right[1]
        sentence = sentence.replace("(", "（")
        sentence = sentence.replace(")", "）")
        # print(sentence)
        try:
            tagged_sent = nlp.posdep(sentence.split(), is_sent=True)
            dep = []
            id2head = {}
            id2word = {0:"root"}
            rel = {}
            tokens = tagged_sent["tokens"]
            for tok in tokens:
                id2word[tok["id"]] = tok["text"]
                id2head[tok["id"]] = tok["head"]
                rel[str(tok["id"]) + str(tok["head"])] = tok["deprel"]
                dep.append([tok["id"], tok["text"], tok["head"], tok["deprel"]])
            dep_path = []
            for d in dep:
                path = [d[1]]
                while id2head[d[0]] != 0:
                    path.append(rel[str(d[0])+str(d[2])])
                    d = dep[d[2]-1]
                path.append("root")
                path.reverse()
                dep_path.append(path)
            
            sentence_tag_path = dep_path
            text = sentence.split()
            # 句法切词和空格切词可能不一致
            path_dic = {}
            flag_dic = {}
            flag_count_dic = {}
            flag_count_use_dic = {}
            # 句子原本词的个数
            
            for w in text:
                value = flag_dic.get(w, "None")
                if value == "None":
                    flag_dic[w] = 1
                    flag_count_dic[w] = 1
                    flag_count_use_dic[w] = 1
                else:
                    flag_dic[w] += 1
            for path in sentence_tag_path:
                value = path_dic.get(path[-1], "None")
                value_n = flag_dic.get(path[-1], 0)
                if value_n == 1:
                    path_dic[path[-1]] = path
                elif value_n > 1:
                    path_dic[path[-1]+str(flag_count_dic[path[-1]])] = path
                    flag_count_dic[path[-1]] += 1

            complete_path = []
            for w in text:
                if flag_dic[w] == 1:
                    complete_path.append(path_dic.get(w, []))
                else:
                    complete_path.append(path_dic.get(w+str(flag_count_use_dic[w]), []))
                    flag_count_use_dic[w] += 1
            all_tag_path.append(complete_path)
        except:
            text = sentence.split()
            all_tag_path.append([[0] for i in range(len(text))])
        # break
    # print(all_tag_path)
    data = np.array(all_tag_path, dtype=object)
    cons_path = filename + ".dep.trankit.npy"
    # print(all_tag_path)
    np.save(cons_path, data)

root = r"C:\Users\ASUS\OneDrive\桌面\indo-dotGCN\datasets"
for dataset in datasets:
    path = root + "\\" + dataset
    print(path)
    get_dep(path)
    # break