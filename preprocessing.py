from nltk.tokenize import word_tokenize 
from nltk import WordNetLemmatizer,wordnet,pos_tag
from nltk.corpus import stopwords
import string

wl=WordNetLemmatizer()
sw=stopwords.raw("english").split()
vocab={}
def get_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.wordnet.ADV
    else:
        return wordnet.wordnet.NOUN
    
def get_lem(text):
    t=""
    for m in text:
        t+=wl.lemmatize(m[0].lower(),get_tag(m[1]))+" "
    return t.strip()

def preprocess(text):
    text=get_lem(pos_tag(word_tokenize(text)))
    new_text=""
    
    for word in text.split():
        if word not in sw:                #remove stop words
            for j in string.punctuation+"0123456789":  #remove special characters
                word=word.replace(j,"")
            if len(word)<=2:
                continue
            new_text+=word+" "
            if word in vocab.keys():
                vocab[word]+=1            #make a dictionary 
            else:
                vocab[word]=1
    return new_text.strip()    
    