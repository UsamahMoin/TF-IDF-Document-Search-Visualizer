# Data Mining project 1
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math

stemmer = PorterStemmer()
script_directory = os.path.dirname(os.path.abspath(__file__))
print("Code Directory: "+script_directory)
corpusroot = os.path.join(script_directory, 'US_Inaugural_Addresses')
print("Corpus root Directory: "+corpusroot)
documents = {}

for filename in os.listdir(corpusroot):
    if filename.endswith('.txt'):
        try:
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                documents[filename] = file.read().lower()
        except Exception as e:
            print(f"Error caught while reading the file")

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    from tools.build_corpus_index import STOP_WORDS
    stop_words = STOP_WORDS

tokenized_documents = {}
for filename, content in documents.items():
    tokens = tokenizer.tokenize(content)
    tokenized_documents[filename] = [stemmer.stem(token) for token in tokens if token not in stop_words]

def compute_idf(tokenized_docs):
    N, idf_dict = len(tokenized_docs), {}
    for tokens in tokenized_docs.values():
        for token in set(tokens):
            idf_dict[token] = idf_dict.get(token, 0) + 1
    for token, df in idf_dict.items():
        idf_dict[token] = math.log10(N / df)
    return idf_dict

idf_values = compute_idf(tokenized_documents)

def getidf(term):
    return idf_values.get(stemmer.stem(term), -1)

def compute_weights(tokenized_docs, idf_vals):
    weights = {}
    for filename, tokens in tokenized_docs.items():
        tf_idf = {}
        for token in tokens:
            tf = 1 + math.log10(tokens.count(token))
            tf_idf[token] = tf * idf_vals.get(token, 0)
        # Cosine normalization
        norm = math.sqrt(sum([value**2 for value in tf_idf.values()]))
        for token, value in tf_idf.items():
            tf_idf[token] = value / norm
        weights[filename] = tf_idf
    return weights

document_weights = compute_weights(tokenized_documents, idf_values)

def getweight(doc, term):
    return document_weights.get(doc, {}).get(stemmer.stem(term), 0)

def query(q):
    query_tokens = tokenizer.tokenize(q.lower())
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in stop_words]
    query_weights = {}
    for token in query_tokens:
        tf = 1 + math.log10(query_tokens.count(token))
        query_weights[token] = tf
    # Cosine
    norm = math.sqrt(sum([value**2 for value in query_weights.values()]))
    for token, value in query_weights.items():
        query_weights[token] = value / norm
    scores = {}
    for doc, weights in document_weights.items():
        scores[doc] = sum([query_weights.get(token, 0) * weight for token, weight in weights.items()])
    return max(scores, key=scores.get), scores[max(scores, key=scores.get)]

print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('war'))
print("%.12f" % getidf('military'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('02_washington_1793.txt','arrive'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('12_jackson_1833.txt','union'))
print("%.12f" % getweight('09_monroe_1821.txt','british'))
print("%.12f" % getweight('05_jefferson_1805.txt','public'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("false public"))
print("(%s, %.12f)" % query("people institutions"))
print("(%s, %.12f)" % query("violated willingly"))
