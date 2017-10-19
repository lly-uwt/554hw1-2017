import os, pandas, nltk, numpy, string
nltk.download('punkt')

print('Program Start')
fileList = os.listdir('transcripts')
numDocs = len(fileList)
print('Number of files:', numDocs)

df1 = pandas.DataFrame(index=fileList, columns=['#Tokens', 'Tokens'])

strtxt = ""
for x in fileList:
    txt = open('transcripts/' + x, encoding='utf-8-sig').read()
    txt = txt.lower().replace('\ufeff', '') # Get rid of BOM
    strtxt += txt
    tokens = nltk.word_tokenize(txt.translate(str.maketrans('', '', string.punctuation)))
    df1.loc[x] = pandas.Series({'#Tokens': len(tokens), 'Tokens': tokens})

totalToken = nltk.word_tokenize(strtxt.translate(str.maketrans('', '', string.punctuation)))
unique = set(totalToken)

# Counting words' frequency
count = {}
for word in totalToken:
    try:
      count[word] += 1
    except KeyError:
      count[word] = 1

# If the word's frequency == 1 -> occur once
once = 0
for word in count:
    if(count[word] == 1):
      once += 1

df2 = pandas.DataFrame.from_dict(count, orient='index').rename(columns={0: 'TF'})  # TF is #Count
df2.sort_values(['TF'], ascending=False, inplace=True)
df2 = df2.iloc[:30] # top 30

df2['N'] = numDocs
df2['DF'] = 0

# Checking word appearance in documents
for word in df2.index.values:
    for tokens in df1['Tokens']:
        if word in tokens:
            df2.loc[word, 'DF'] += 1

df2['IDF'] = numpy.log(df2['N'].divide(df2['DF'])) # log(N/DF)

df2['TF*IDF'] = df2['TF'] * df2['IDF'] # TF*IDF
df2['Probability'] = df2['TF'].divide(len(totalToken)) # TF/totalToken

print('The number of word tokens in the database:', len(totalToken))
print('The number of unique words in the database:', len(unique))
print('The number of words that occur only once in the database:', once)
print('For 30 most frequent words in the database, provide: TF, IDF, TF*IDF and probabilities')
print(df2)
print('The average number of word tokens per document:',  len(totalToken) / 404)
print('Program Ended')
