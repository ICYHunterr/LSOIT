# conda activate D:\anaconda\envs\const
import stanza
nlp = stanza.Pipeline('en')
# air has higher resolution but the fonts are small .
doc = nlp('air has higher resolution but the fonts are small .')
dep = []
sent = doc.sentences[0]
id2head = {}
id2word = {0:"root"}
rel = {}
for word in sent.words:
    id2word[word.id] = word.text
    id2head[word.id] = word.head
    rel[str(word.id) + str(word.head)] = word.deprel
    dep.append([word.id, word.text, word.head, sent.words[word.head-1].text if word.head > 0 else "root", word.deprel])
dep_path = []
for d in dep:
    path = [d[1]]
    while id2head[d[0]] != 0:
        path.append(rel[str(d[0])+str(d[2])])
        d = dep[d[2]-1]
    path.append("root")
    path.reverse()
    dep_path.append(path)
    
# print(dep)
# print(id2head)
# print(id2word)
# print(rel)
print(dep_path)