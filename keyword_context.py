import pandas as pd
from nltk import pos_tag
#nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Read the file
df = pd.read_csv('Glassdoor company review by Xiaolei0827.csv',index_col=0)
df = df.dropna()

keywords_filter = ['culture','value','philosophy','belief']

# Merge Columns "pros", "cons", and "advice to management" in column text
df['text'] = df[df.columns[14:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

df['text_cleaned'] = df['text'].map(lambda x: x.lower())

lemmatizer = WordNetLemmatizer()

#def stem(text):
#    txt = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else lemmatizer.lemmatize(i) for i,j in pos_tag(word_tokenize(text))]
#    print(txt)

df['contained'] = df['text_cleaned'].apply(lambda x: 1 if any(s in lemmatizer.lemmatize(x)for s in keywords_filter) else 0)

df_filter = df[df['contained']==1]

#df_filter.contained.sum() # 62,006 reviews

df_filter['review_sentences'] = df_filter['text_cleaned'].apply(lambda text: [sent for sent in sent_tokenize(text)
                                       if any(True for w in word_tokenize(sent)
                                       if lemmatizer.lemmatize(w).lower() in keywords_filter)])

# count the number of sentences in each review
df_filter['num_sent'] = df_filter['text_cleaned'].apply(lambda x: len(sent_tokenize(x)))

df_filter = df_filter.reset_index(drop=True)

tokenized_sents = [sent_tokenize(i) for i in df_filter.text_cleaned]

sent = []
for i in list(range(df_filter.shape[0])):
    for j in list(range(df_filter.num_sent[i])):

        if len(list(range(df_filter.num_sent[i]))) < 3:
            x = df_filter.review_sentences[i]
            if x not in sent:
            #x = tokenized_sents[i][j]
                sent.append(x)

        elif j > 0 and tokenized_sents[i][j] in df_filter.review_sentences[i]:
            y = [tokenized_sents[i][j-1],tokenized_sents[i][j],tokenized_sents[i][j+1]]
            sent.append(y)

        elif tokenized_sents[i][j] in df_filter.review_sentences[i]:
            z = [tokenized_sents[i][j],tokenized_sents[i][j+1]]
            sent.append(z)
        else:
            continue

#sent_index = []
#index = []
temp_sum = pd.Series([])
for i in list(range(df_filter.shape[0])):
    index_test = []
    sent_index = []
    print(i)
    print("----------")
    print("this is " + str(i) +"th row")
    print("----------")
    temp_num_sent = df_filter["num_sent"][i]
    # this is the indexed sentence
    series = sent_tokenize(df_filter.text_cleaned[i])
    #print("this is series: ")
    #print(series)

    for k in list(range(df_filter.keyword_freq[i])):
        lookup_sentence = df_filter['review_sentences'][i][k]
        #print(lookup_sentence)
        x = pd.Index(series).get_loc(lookup_sentence)
        #print("this is x: " + str(x))
        index_tuples = (lookup_sentence,x)
        index_3 = [x-1,x,x+1]
        #print("this is series x: ")
        #print(series[x])
        value_tuples = series[x-1:x+2]

        #for item in index_3:

        if index_tuples not in sent_index:
        #if value_tuples not in sent_index:
            #print(index_tuples)
            #print(index_3)
            #index.append(index_3)
            index_test.append(index_3)
            #print(index)
            #print(Union(index[i],index_3))
            #print(remove_duplicates(index))


            #print(value_tuples)



            #[series[x-1:x+2]]
            sent_index.append(index_tuples)
            #print(value_tuples)
        else:
            pass
    index_test = sum(index_test, [])
    index_test = list(set(index_test))

    try:
        index_test.remove(-1)
    except ValueError:
        pass
    try:
        index_test.remove(temp_num_sent)
    except ValueError:
        pass
    print("----------")
    #print("this is index: ")
    print("this is index_test: ")
    print(index_test)
    #print("this is sent_index: ")
    #print(sent_index)
    #print("----------")
    #print("          ")
    temp_out = [series[i] for i in index_test]
    print("this is temp_out: ")
    print(temp_out)
    temp_sum = temp_sum.append(pd.Series([temp_out]), ignore_index=True)
    print("this is temp_sum")
    print(temp_sum)



#df_filter['keyword_3context'] = sent

#df_filter.to_csv('df_filter.csv', index=False)
