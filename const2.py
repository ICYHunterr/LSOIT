# conda activate D:\anaconda\envs\const
from supar import Parser
from nltk import Tree
con = Parser.load('crf-con-en')

import numpy as np
datasets = ["semeval14\\laptop_test.raw","semeval14\\laptop_train.raw","semeval15\\restaurant_train.raw", "semeval15\\restaurant_test.raw"]
def get_const(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    all_tag_path = []
    for i in range(0, len(lines), 3):
        left_and_right = lines[i].strip().split("$T$")
        sentence = left_and_right[0] +lines[i+1].strip() + left_and_right[1]
        sentence = sentence.replace("(", "（")
        sentence = sentence.replace(")", "）")
        try:
            const = con.predict(sentence.split(), verbose=False)[0]
            t = Tree.fromstring(str(const))  
            leaves_pos = t.treepositions('leaves')
            sentence_tag_path = []
            for leaves in leaves_pos:
                word_tag_path = [t[leaves]]
                del_n = len(leaves) - 1
                for i in range(1, len(leaves)):
                    if len(leaves[:-i]) == del_n:
                        continue
                    word_tag_path.append(t[leaves[:-i]].label())
                word_tag_path.reverse()
                sentence_tag_path.append(word_tag_path)

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
        # print(all_tag_path)
    data = np.array(all_tag_path, dtype=object)
    cons_path = filename + ".crf-con.const.npy"
    # print(all_tag_path)
    np.save(cons_path, data)
root = r"C:\Users\ASUS\OneDrive\桌面\indo-dotGCN\datasets"
for dataset in datasets:
    path = root + "\\" + dataset
    print(path)
    get_const(path)