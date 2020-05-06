# Import modules
import numpy as np
import pandas as pd
import re
import spacy
import tqdm
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis.gensim
import pickle
import pyLDAvis
from pprint import pprint

''''''''''''''''''''''''''''''''''
Data Loading
''''''''''''''''''''''''''''''''''
df = pd.read_csv('reviewproscons.csv',index_col=0)
#df.head()

''''''''''''''''''''''''''''''''''
Data Cleaning
''''''''''''''''''''''''''''''''''
# Remove the nulls
df = df.dropna()
# Remove punctuation
df['review_cleaned'] = df['review'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the reviews to lowercase
df['review_cleaned'] = df['review_cleaned'].map(lambda x: x.lower())
# Print out the first rows of reviews
#df['review_cleaned'].head()

''''''''''''''''''''''''''''''''''
Keywords filter
''''''''''''''''''''''''''''''''''
keywords_filter = ['culture','value','philosophy','belief']
df['contained'] = df['review'].apply(lambda x: 1 if any(s in x for s in keywords_filter) else 0)
df_filter = df[df['contained']==1]


''''''''''''''''''''''''''''''''''
Tokenization
''''''''''''''''''''''''''''''''''
#Tokenize words and further clean-up text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=False))  # deacc=True removes punctuations
data = df_filter.review_cleaned.values.tolist()
data_words = list(sent_to_words(data))
#print(data_words[:1])

''''''''''''''''''''''''''''''''''
Bigrams and Trigrams
''''''''''''''''''''''''''''''''''
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
# See trigram example
#print(trigram_mod[bigram_mod[data_words[5]]])

# NLTK Stop words
# import nltk
# nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#print(data_lemmatized[:1])

'''''''''''''''''''''''''''''''''''''''''''''
Data Transformation: Corpus and Dictionary
'''''''''''''''''''''''''''''''''''''''''''''
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

'''''''''''''''''''''''''''''''''''''''''''''
Build LDA model
'''''''''''''''''''''''''''''''''''''''''''''
# Build LDA model
lda_model = gensim.models.LdaMulticore(workers=3,
                                       corpus=corpus,
                                       id2word=id2word,
                                       num_topics=10,
                                       random_state=100,
                                       chunksize=100, # number of documents to be used in each training chunk
                                       passes=10,     # total number of training passes
                                       per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

'''''''''''''''''''''''''''''''''''''''''''''
Compute Model Perplexity and Coherence Score¶
'''''''''''''''''''''''''''''''''''''''''''''
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b,
                                           per_word_topics=True)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'lda_gensim.html')
vis
