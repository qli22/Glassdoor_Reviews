import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
# load data - change your dataset name here
df = pd.read_csv('reviewproscons.csv',index_col=0)
# Remove the nulls
df = df.dropna()
# Remove punctuation - don't remove period "." now
df['review_cleaned'] = df['review'].map(lambda x: re.sub('[,\!?]', '', x))
# Convert the reviews to lowercase
df['review_cleaned'] = df['review_cleaned'].map(lambda x: x.lower())

# Create keyword filter
keywords_filter = ['culture','value','philosophy','belief']
# Filter out the reviews contain the keywords
df['contained'] = df['review'].apply(lambda x: 1 if any(s in x for s in keywords_filter) else 0)
df_filter = df[df['contained']==1]
# Print the number of reviews those contain keywords
df_filter.shape

# Filter out the sentences containing keywords from entire review text
review_sentence = df_filter['review_cleaned'].apply(lambda text: [sent for sent in sent_tokenize(text)
                                       if any(True for w in word_tokenize(sent)
                                               if w.lower() in keywords_filter)])
# Remove empty lists
#review_sentence = review_sentence[review_sentence.astype(str) != '[]']

# Print the number of sentences those contain keywords
review_sentence.shape
