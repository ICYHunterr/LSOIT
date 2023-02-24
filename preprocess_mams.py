# encodings:utf-8

import numpy as np
import spacy
import pickle
import stanza
import sys

#nlp = spacy.load('en_core_web_sm')
nlp = stanza.Pipeline('en', processors='tokenize', tokenize_no_ssplit=True) # initialize English neural pipeline
output_file = open(sys.argv[1]+".new", "w")
lines = [line for line in open(sys.argv[1], "r")]
i = 0
while i < len(lines):
    #print(line)
    original_line = lines[i].strip()
    #print("[tlog] ori: " + original_line)

    text_left, aspect,  text_right = [s.strip() for s in lines[i].partition("$T$")]
    new_line = ""
    if len(text_left)>0:
        tokenized_left = []
        text_left = nlp(text_left)
        for sent in text_left.sentences:
            tokenized_left.extend([word.text for word in sent.words])

        new_line = " ".join(tokenized_left)

    new_line = new_line + " $T$ " 
    if len(text_right)>0:
        text_right = nlp(text_right)
        tokenized_right = [] 
        for sentence in text_right.sentences:
        #print(text_left)
            tokenized_right = [word.text for word in sentence.words]

        new_line = new_line  + " ".join(tokenized_right)

    new_line = new_line.strip()
    
    words1 = "".join(original_line.split())
    words2 = "".join(new_line.split())
    if words1 != words2:
        print("[tlog] w1: " + words1)
        print("[tlog] w2: " + words2)
        sys.exit(0)
    assert words1 == words2
    #print(words1)
    #print(words2)
    #print(new_line)
    output_file.write(new_line + "\n")
    output_file.write(lines[i+1])
    output_file.write(lines[i+2])
    #sys.exit(0)
    i +=3
