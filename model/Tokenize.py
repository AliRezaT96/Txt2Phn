import re

class tokenize(object):
    
    def __init__(self):
        pass
            
    def WordTokenizer(self, word):
        word = word.lower()
        return word.split("")
    
    def IPATokenizer(self, IPA):
        return IPA.split("")
