
# coding: utf-8

# In[ ]:

import re
import string
# spacy for lemmatization
import spacy

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), 
                        (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), 
                        (r'(\w+)n\'t', '\g<1> not'),(r'(\w+)\'ve', '\g<1> have'), 
                        (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), 
                        (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), 
                        (r'dont', 'do not'), (r'wont', 'will not') ]
def replace_contraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text


PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'us', 'seems','still','could','would','already','everything','yet','really','nan','usual',
                   'yes','always','overall','mr','guy','sure','lady','etc','miss','somebody','many','say',
                   'everyone','due','somebody','want','also',
                   'thank','thanks','much','able','please','can','nothing','none','well'])
words_keep = ['against','own','same','rare','without'] #'all','no','not',
#words_keep = ['against','own','same','rare','without'] #'all','no','not',
stop_words = [word for word in stop_words if word not in words_keep]

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in stop_words])



sp = spacy.load("en_core_web_sm", disable=['tagger','ner', 'parser'])
sp.add_pipe(sp.create_pipe('sentencizer'))
def get_lemmatized_text(corpus, sp):
    if len(corpus) > 0:
        doc = sp(corpus)
        output = " ".join([token.lemma_.lower() if token.lemma_ != '-PRON-' else token.lower_ for token in doc])
    else:
        output = corpus
    return output


from nltk.corpus import wordnet

def replace(word, pos=None):
    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None

def replaceNegations(text):
    """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
    i, l = 0, len(text)
    words = []
    while i < l:
        word = text[i]
        if word == 'not' and i+1 < l:
            ant = replace(text[i+1])
            if ant:
                words.append(ant)
                i += 2
                continue
        words.append(word)
        i += 1
    return words

def tokenize1(text):
    tokens = nltk.word_tokenize(text)
    tokens = replaceNegations(tokens)
    text = " ".join(tokens)
    return text




def pre_processing(text):
    #lowercase & remove extra whitespace
    text = " ".join(text.lower().split()) 
    
    #remove urls, htmls, emotj
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_emoji(text)

    #replace contraction - expand
    text = replace_contraction(text)
    
    #remove punctuation
    text = remove_punctuation(text)
    
    # remove stop words
    #text = remove_stopwords(text)
    
    #lemmatization
    #text = get_lemmatized_text(text, sp)
    
    #replace negations with antonym , not happy to unhappy
    #text = tokenize1(text)
    #text = remove_stopwords(text)
    
    return text

from collections import Counter
def word_count(n_freq, n_rare, text_list):
    cnt_word = Counter()
    for text in text_list:
        unique_text = set(text.split(' '))
        for word in unique_text:
    #    for word in text.split():   #count frequency of word in all documents, duplicated occurancy would be included
            cnt_word[word] += 1

    FREQWORDS = Counter(el for el in cnt_word.elements() if cnt_word[el] >= n_freq).keys()
    RAREWORDS = Counter(el for el in cnt_word.elements() if cnt_word[el] <= n_rare).keys()
    WORDS = Counter(el for el in cnt_word.elements() if (cnt_word[el] >= n_freq) | (cnt_word[el] <= n_rare)).keys()

    return WORDS

def remove_frequency_words(text, WORDS):
    
    #WORDS = word_count(n_freq, n_rare, text_list)
    """custom function to remove the rare words"""
    #return " ".join([word for word in str(text).split() if word not in RAREWORDS])
    return " ".join([word for word in str(text).split() if word not in WORDS])

