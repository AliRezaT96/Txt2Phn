import pandas as pd
import torchtext
import torch
from torchtext.legacy import data
from model.Batch import MyIterator, batch_size_fn
import os
import dill as pickle

 
def WordTokenizer(word):
    return list(word.lower())
    
def IPATokenizer(IPA):
    return list(IPA)


def read_data(opt):
    
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    
    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    
    #spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl']
    #if opt.src_lang not in spacy_langs:
     #   print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)  
    #if opt.trg_lang not in spacy_langs:
       # print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)

    TRG = data.Field(lower=True, tokenize=WordTokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=False, tokenize=IPATokenizer)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return(SRC, TRG)

def create_dataset(opt, SRC, TRG):

    print("creating dataset and iterator... ")

    raw_data = {'src' : [' '.join(list(line.split('\t')[0])) for line in opt.src_data], 'trg': [' '.join(list(line.split('/')[1])) for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]
    print(df)

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device='cpu',
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir("weights")
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
            pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

            with open('SRCVocab.txt', 'w') as f:
                for item in list(SRC.vocab.freqs):
                  f.write("%s\n" % item)

            with open('TRGVocab.txt', 'w') as f:
                for item in list(TRG.vocab.freqs):
                  f.write("%s\n" % item)

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    #opt.train_len = get_len(train_iter)
    opt.train_len = int(len(df)/opt.batchsize)

    return train_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i
