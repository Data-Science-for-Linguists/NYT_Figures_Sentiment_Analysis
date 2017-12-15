import pandas as pd
import pickle
from pprint import pprint

# This performs a simple analysis on a small subset of the data

# Load in pickled "unpacked" file that was created from unpack_year_of_data.py
pd2007 = pickle.load(open("../jupyter_notebooks/nyt-2007.p", "rb"))

# Group everything by Name and DOCID and save it to a new dataframe
pd2007_sorted = pd2007.groupby(['Name', 'DOCID']).count()

# Example - Use the first person to be mentioned in multiple articles
pd2007_sorted.xs("AARON, HANK").index.values.tolist()

# Grab a list of all the names in the file
names_2007 = list(set(pd2007["Name"]))

names_and_occurences = {}
for name in names_2007:
    names_and_occurences[name] = pd2007_sorted.xs(name).index.values.tolist()

pprint(names_and_occurences)

sent_scores = pd.DataFrame(columns=['Name', 'Num_Docs', 'NEG', 'NEU', 'POS'])
for name in names_and_occurences:
    numdocs = len(names_and_occurences.get(name))
    neg = []
    neu = []
    pos = []
    for doc in names_and_occurences.get(name):
        
