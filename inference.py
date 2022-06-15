import argparse
import time
import torch
from model.Models import get_model
from preprocess import *
import torch.nn.functional as F
from model.Optimizer import CosineWithRestarts
from model.Batch import create_masks
import pdb
import dill as pickle
import argparse
from model.Beam import beam_search
from torch.autograd import Variable
import re

def translate_sentence(word, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(word)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(tok)
    sentence = Variable(torch.LongTensor([indexed]))
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return  ''.join(sentence.split())

def translate(opt, model, SRC, TRG):
    words = opt.text.lower()
    translated = []

    for word in words.split():
        translated.append(translate_sentence(word, model, opt, SRC, TRG))

    return (' '.join(translated))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()
 
    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    
    while True:
        opt.text =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='f':
            fpath =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opt, model, SRC, TRG)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
