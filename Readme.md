## Fake News Detector -- Check releases for the script file
ML model that detects fake news. Logistic regression and recurrent neural networks are used comparatively, with both acheiving around 98% accuracy. More information is provided below along with the datasets.
<hr>
```python
import pydot
import graphviz
```

## 1. importing the neccessary libraries


```python
from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import inflect
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
```

## 2. Read in the data


```python
# Read the data from the CSV file into a pandas DataFrame
real_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')
```

## 3. Exploratory Data Analysis

#### 3.1 exploring the real news dataset


```python
#printing info for the real news dataset
display(Markdown("**The information about the real news dataset is as follows:**"))
print(real_news.info())
```


**The information about the real news dataset is as follows:**


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21417 entries, 0 to 21416
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   title    21417 non-null  object
     1   text     21417 non-null  object
     2   subject  21417 non-null  object
     3   date     21417 non-null  object
    dtypes: object(4)
    memory usage: 669.4+ KB
    None
    


```python
# lets check the 5 first rows of the real news dataset
display(Markdown("**The first five rows of the real news dataset are as follows:**"))
print(real_news.head())
```


**The first five rows of the real news dataset are as follows:**


                                                   title  \
    0  As U.S. budget fight looms, Republicans flip t...   
    1  U.S. military to accept transgender recruits o...   
    2  Senior U.S. Republican senator: 'Let Mr. Muell...   
    3  FBI Russia probe helped by Australian diplomat...   
    4  Trump wants Postal Service to charge 'much mor...   
    
                                                    text       subject  \
    0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   
    1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   
    2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   
    3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   
    4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   
    
                     date  
    0  December 31, 2017   
    1  December 29, 2017   
    2  December 31, 2017   
    3  December 30, 2017   
    4  December 29, 2017   
    

#### 3.2 exploring the fake news dataset


```python
# lets check the 5 first rows of the real news dataset
display(Markdown("**The first five rows of the fake news dataset are as follows:**"))
print(fake_news.info())
```


**The first five rows of the fake news dataset are as follows:**


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23481 entries, 0 to 23480
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   title    23481 non-null  object
     1   text     23481 non-null  object
     2   subject  23481 non-null  object
     3   date     23481 non-null  object
    dtypes: object(4)
    memory usage: 733.9+ KB
    None
    


```python
#printing info for the real news dataset
display(Markdown("**The information about the fake news dataset is as follows:**"))
print(fake_news.info())
```


**The information about the fake news dataset is as follows:**


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23481 entries, 0 to 23480
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   title    23481 non-null  object
     1   text     23481 non-null  object
     2   subject  23481 non-null  object
     3   date     23481 non-null  object
    dtypes: object(4)
    memory usage: 733.9+ KB
    None
    

#### Looking at the data information, we can see that the real dataset has 21417 rows and the fake dataset has 23481 rows. 

#### both datasets have the same number of columns. which are title, text, subject, date.

#### we can also see that there are no labels in the dataset as the lable is indicated in the file name.

#### Therefore, we will add the labels to the dataset

#### 3.3 prepare the 2 dataframes for merging


```python
#add label indicating real or fake labels before combining
real_news['label'] = 0
fake_news['label'] = 1
```

#### 3.4  Combine the two dataframes into a single dataframe and shuffle the rows of the combined dataframe.


```python
# combine the two dataset rea_news and fake_news into one dataset called df
df = pd.concat([real_news, fake_news])

# Shuffle the DataFrame to randomize the order of the rows
df = df.sample(frac=1).reset_index(drop=True)

```


```python
df.to_csv('processed_data.csv', index=False)
```


```python
# print the info of the 1st 15 rows of the dataset
df.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>subject</th>
      <th>date</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>While Her Department Gets Gutted, Billionaire...</td>
      <td>Donald Trump and his administration seem to be...</td>
      <td>News</td>
      <td>April 10, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Watergate 'pales' compared with Trump-Russia: ...</td>
      <td>SYDNEY (Reuters) - The Watergate scandal pales...</td>
      <td>politicsNews</td>
      <td>June 7, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hillary Just Showed The Whole World That She ...</td>
      <td>Following Covfefe-gate, in which Trump made an...</td>
      <td>News</td>
      <td>June 1, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Treasury secretary's wife apologizes for Insta...</td>
      <td>(Reuters) - The wife of U.S. Treasury Secretar...</td>
      <td>politicsNews</td>
      <td>August 22, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Exclusive: U.S. needs to improve oversight of ...</td>
      <td>CHICAGO (Reuters) - A year-long audit of the p...</td>
      <td>politicsNews</td>
      <td>October 31, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>No cult of personality around Xi, says top Chi...</td>
      <td>BEIJING (Reuters) - China has learnt from hist...</td>
      <td>worldnews</td>
      <td>November 6, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>EU says needs concrete evidence from Turkey to...</td>
      <td>ANKARA (Reuters) - The European Union does not...</td>
      <td>worldnews</td>
      <td>November 30, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>U.S. approves possible $15 billion sale of THA...</td>
      <td>WASHINGTON (Reuters) - The U.S. State Departme...</td>
      <td>worldnews</td>
      <td>October 6, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Speaker Ryan dented by healthcare debacle, but...</td>
      <td>WASHINGTON (Reuters) - U.S. House Speaker Paul...</td>
      <td>politicsNews</td>
      <td>March 25, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>WHERE’S THE OVERSIGHT? OBAMA FUNNELED BILLIONS...</td>
      <td>Advocates for big government and progressive ...</td>
      <td>Government News</td>
      <td>Mar 2, 2017</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>China's precedent-breaking Xi Jinping gets set...</td>
      <td>BEIJING (Reuters) - Chinese President Xi Jinpi...</td>
      <td>worldnews</td>
      <td>October 16, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>France's Macron picked moment to put EU propos...</td>
      <td>PARIS (Reuters) - French President Emmanuel Ma...</td>
      <td>worldnews</td>
      <td>September 26, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Russia says general killed in Syria held senio...</td>
      <td>MOSCOW (Reuters) - A Russian general killed in...</td>
      <td>worldnews</td>
      <td>September 27, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Trump proposes U.S. tax overhaul, stirs concer...</td>
      <td>WASHINGTON (Reuters) - President Donald Trump ...</td>
      <td>politicsNews</td>
      <td>September 27, 2017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Beijing detains 18 after apartment fire: Xinhua</td>
      <td>SHANGHAI (Reuters) - Police in Beijing have de...</td>
      <td>worldnews</td>
      <td>November 21, 2017</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the number of instances (rows) and features (columns)
num_instances, num_features = df.shape
print("Number of instances (rows):", num_instances)
print("Number of features (columns):", num_features)
```

    Number of instances (rows): 44898
    Number of features (columns): 5
    


```python
# data types of each column in the dataset
df.dtypes
```




    title      object
    text       object
    subject    object
    date       object
    label       int64
    dtype: object



#### 3.5 Data Vizualization


```python
# plot a pie chart the shows  the percentage of fake and real news in the dataset


# Count the number of real news and fake news instances
num_real_news = (df['label'] == 0).sum()  #  0 represents real news
num_fake_news = (df['label'] == 1).sum()  #  1 represents fake news

print(f'Number of real news instances: {num_real_news}')
print(f'Number of fake news instances: {num_fake_news}')

# Create a new figure object
fig = plt.figure(figsize=(8, 8))

# Plot a pie chart
explode = (0.1, 0)  # Explode the first slice (real news) slightly
colors = ['#ff9999', '#66b3ff']  # Custom colors
labels = ['Fake News', 'Real News']  # Custom labels
df['label'].value_counts().plot(kind='pie', autopct='%1.0f%%', explode=explode, colors=colors, labels=labels)

# Add title
plt.title('Percentage of Fake and Real News in the Dataset')

# Remove y-axis label
plt.ylabel('')  

# Show the plot
plt.show()
```

    Number of real news instances: 21417
    Number of fake news instances: 23481
    


    
![png](https://i.imgur.com/WmXqiyW.png)
    


## 4 Data Preprocessing

#### 4.1 Checking for missing and duplicate values


```python
# Check for null instances in the DataFrame
df = df[~df['title'].str.contains('Video', case=False)]
df = df[df['text'].str.strip() != '']
null_instances = df.text.isnull().any()
duplicates = df.duplicated().any()  # Check for duplicates
print("Null instances in the DataFrame:\n"  , null_instances)
print("Duplicates in the DataFrame:\n" , duplicates)
df.drop_duplicates(inplace=True)  # Drop duplicates
df.dropna(inplace=True)  # Drop null instances
```

    Null instances in the DataFrame:
     False
    Duplicates in the DataFrame:
     True
    


```python
# Count the number of real news and fake news instances
num_real_news = (df['label'] == 0).sum()  #  0 represents real news
num_fake_news = (df['label'] == 1).sum()  #  1 represents fake news

print(f'Number of real news instances: {num_real_news}')
print(f'Number of fake news instances: {num_fake_news}')
```

    Number of real news instances: 21170
    Number of fake news instances: 14998
    

#### 4.2 Droping the columes that are not required for the model


```python
# drop the title, subject, and date columns
df_v2 = df.drop(['title', 'subject', 'date'], axis=1)  
```


```python
display(Markdown("**The information about the dataset is as follows:**"))
print(df_v2.head(20))
```


**The information about the dataset is as follows:**


                                                     text  label
    0   Donald Trump and his administration seem to be...      1
    1   SYDNEY (Reuters) - The Watergate scandal pales...      0
    2   Following Covfefe-gate, in which Trump made an...      1
    3   (Reuters) - The wife of U.S. Treasury Secretar...      0
    4   CHICAGO (Reuters) - A year-long audit of the p...      0
    5   BEIJING (Reuters) - China has learnt from hist...      0
    6   ANKARA (Reuters) - The European Union does not...      0
    7   WASHINGTON (Reuters) - The U.S. State Departme...      0
    8   WASHINGTON (Reuters) - U.S. House Speaker Paul...      0
    10  BEIJING (Reuters) - Chinese President Xi Jinpi...      0
    11  PARIS (Reuters) - French President Emmanuel Ma...      0
    12  MOSCOW (Reuters) - A Russian general killed in...      0
    13  WASHINGTON (Reuters) - President Donald Trump ...      0
    14  SHANGHAI (Reuters) - Police in Beijing have de...      0
    15  LONDON (Reuters) - Nigel Farage, an anti-immig...      0
    16  SEOUL/WASHINGTON (Reuters) - Two U.S. strategi...      0
    18  WASHINGTON (Reuters) - The top Democrat on the...      0
    19  WASHINGTON (Reuters) - White House Economic Ad...      0
    21  SAN FRANCISCO (Reuters) - Apple Inc, Alphabet ...      0
    24  WASHINGTON (Reuters) - Republican presidential...      0
    

#### 4.3 Data Cleaning

##### 4.3.1 converting numercial values to words


```python
#  use the inflect library inflect, which provides facilities for converting numbers to words in English.

from inflect import engine
import re

# Initialize inflect engine
p = engine()

# Define a function to convert alphanumeric sequences containing digits to their string representations
def replace_numeric_words(text):
    return re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group()), text)
```


```python
df_v2['text'] = df_v2['text'].apply(replace_numeric_words)  # Apply the function to the 'text' column
```


```python
df_v2.to_csv('processed_data1.csv', index=False)
```


```python
display(Markdown("**The first 20 rows of the dataset are as follows:**"))
print(df_v2.head(20))
```


**The first 20 rows of the dataset are as follows:**


                                                     text  label
    0   Donald Trump and his administration seem to be...      1
    1   SYDNEY (Reuters) - The Watergate scandal pales...      0
    2   Following Covfefe-gate, in which Trump made an...      1
    3   (Reuters) - The wife of U.S. Treasury Secretar...      0
    4   CHICAGO (Reuters) - A year-long audit of the p...      0
    5   BEIJING (Reuters) - China has learnt from hist...      0
    6   ANKARA (Reuters) - The European Union does not...      0
    7   WASHINGTON (Reuters) - The U.S. State Departme...      0
    8   WASHINGTON (Reuters) - U.S. House Speaker Paul...      0
    10  BEIJING (Reuters) - Chinese President Xi Jinpi...      0
    11  PARIS (Reuters) - French President Emmanuel Ma...      0
    12  MOSCOW (Reuters) - A Russian general killed in...      0
    13  WASHINGTON (Reuters) - President Donald Trump ...      0
    14  SHANGHAI (Reuters) - Police in Beijing have de...      0
    15  LONDON (Reuters) - Nigel Farage, an anti-immig...      0
    16  SEOUL/WASHINGTON (Reuters) - Two U.S. strategi...      0
    18  WASHINGTON (Reuters) - The top Democrat on the...      0
    19  WASHINGTON (Reuters) - White House Economic Ad...      0
    21  SAN FRANCISCO (Reuters) - Apple Inc, Alphabet ...      0
    24  WASHINGTON (Reuters) - Republican presidential...      0
    

##### 4.3.2 removing  punctuations and all link and tags and  converting all text to lower case


```python
#defining a function to clean the text
def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove quotes
    text = text.replace("'", "").replace('"', '').replace("``", '').replace("''", '')

    # Convert text to lower case to ensure consistency
    text = text.lower()

    # Remove text in square brackets
    text = re.sub('\[.*?\]', '', text)

    # Remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Define a regular expression pattern to match words containing both letters and digits
    text = re.sub(r'\b(?=\w*\d)(?=\w*[a-zA-Z])\w+\b', ' ', text)

    # Remove punctuation using the string.punctuation constant
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    # Replace new line characters with a single space
    text = re.sub('\n', ' ', text)
    
    # Remove single leters
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text
```


```python
df_v3 = df_v2.copy()
df_v3['text'] = df_v3['text'].apply(clean_text)
```


```python
df_v3.to_csv('processed_data2.csv', index=False)
```


```python
display(Markdown("**The first 20 rows of the dataset are as follows:**"))
print(df_v3.head(20))
```


**The first 20 rows of the dataset are as follows:**


                                                     text  label
    0   donald trump and his administration seem to be...      1
    1   sydney reuters the watergate scandal pales in ...      0
    2   following covfefe gate in which trump made an ...      1
    3   reuters the wife of treasury secretary steve m...      0
    4   chicago reuters year long audit of the program...      0
    5   beijing reuters china has learnt from history ...      0
    6   ankara reuters the european union does not sha...      0
    7   washington reuters the state department has ap...      0
    8   washington reuters house speaker paul ryan on ...      0
    10  beijing reuters chinese president xi jinping i...      0
    11  paris reuters french president emmanuel macron...      0
    12  moscow reuters russian general killed in syria...      0
    13  washington reuters president donald trump prop...      0
    14  shanghai reuters police in beijing have detain...      0
    15  london reuters nigel farage an anti immigratio...      0
    16  seoul washington reuters two strategic bombers...      0
    18  washington reuters the top democrat on the sen...      0
    19  washington reuters white house economic advise...      0
    21  san francisco reuters apple inc alphabet inc g...      0
    24  washington reuters republican presidential can...      0
    

##### 4.4 Tokenization

#### 4.4.1 Removing stop words


```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load English stopwords
stop = stopwords.words('english')


# Remove stop words from the 'text' column
df_v3['text_without_stopwords'] = df_v3['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

```

##### 4.4.2 Tokanization


```python
# Tokenize the 'text_without_stopwords' column using word_tokenize
df_v3['tokenized_text'] = df_v3['text_without_stopwords'].apply(word_tokenize)
```

##### 4.4.3 Lemmatization


```python
import nltk
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# apply lemmatization
lemmatizer = WordNetLemmatizer()


def lemmatize_with_pos(text):
    #tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(text)  # Perform POS tagging
    lemmatized_tokens = []
    for token, tag in tagged_tokens:
        # Map POS tags to WordNet POS tags
        wn_tag = wordnet.ADJ if tag.startswith('J') else \
                 wordnet.VERB if tag.startswith('V') else \
                 wordnet.NOUN if tag.startswith('N') else \
                 wordnet.ADV if tag.startswith('R') else \
                 None
        if wn_tag:
            lemmatized_token = lemmatizer.lemmatize(token, pos=wn_tag)
        else:
            lemmatized_token = lemmatizer.lemmatize(token)  # Use default POS tag
        lemmatized_tokens.append(lemmatized_token)
    return ' '.join(lemmatized_tokens)

df_v3['lemmatized_text'] = df_v3['tokenized_text'].apply(lemmatize_with_pos)
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\Kero\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\Kero\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\Kero\AppData\Roaming\nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    


```python
# Lemmatize the tokenized text
#df_v3['lemmatized_text'] = df_v3['tokenized_text'].apply(lambda lst: [lmtzr.lemmatize(word) for word in lst])

```


```python
df_v3.to_csv('processed_data3.csv', index=False)
```


```python
df_v3.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>text_without_stopwords</th>
      <th>tokenized_text</th>
      <th>lemmatized_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>donald trump and his administration seem to be...</td>
      <td>1</td>
      <td>donald trump administration seem austerity lon...</td>
      <td>[donald, trump, administration, seem, austerit...</td>
      <td>donald trump administration seem austerity lon...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sydney reuters the watergate scandal pales in ...</td>
      <td>0</td>
      <td>sydney reuters watergate scandal pales compari...</td>
      <td>[sydney, reuters, watergate, scandal, pales, c...</td>
      <td>sydney reuters watergate scandal pale comparis...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>following covfefe gate in which trump made an ...</td>
      <td>1</td>
      <td>following covfefe gate trump made absolute foo...</td>
      <td>[following, covfefe, gate, trump, made, absolu...</td>
      <td>follow covfefe gate trump make absolute fool s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>reuters the wife of treasury secretary steve m...</td>
      <td>0</td>
      <td>reuters wife treasury secretary steve mnuchin ...</td>
      <td>[reuters, wife, treasury, secretary, steve, mn...</td>
      <td>reuters wife treasury secretary steve mnuchin ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>chicago reuters year long audit of the program...</td>
      <td>0</td>
      <td>chicago reuters year long audit program overse...</td>
      <td>[chicago, reuters, year, long, audit, program,...</td>
      <td>chicago reuters year long audit program overse...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>beijing reuters china has learnt from history ...</td>
      <td>0</td>
      <td>beijing reuters china learnt history allow mao...</td>
      <td>[beijing, reuters, china, learnt, history, all...</td>
      <td>beijing reuters china learnt history allow mao...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ankara reuters the european union does not sha...</td>
      <td>0</td>
      <td>ankara reuters european union share turkey vie...</td>
      <td>[ankara, reuters, european, union, share, turk...</td>
      <td>ankara reuters european union share turkey vie...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>washington reuters the state department has ap...</td>
      <td>0</td>
      <td>washington reuters state department approved p...</td>
      <td>[washington, reuters, state, department, appro...</td>
      <td>washington reuters state department approve po...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>washington reuters house speaker paul ryan on ...</td>
      <td>0</td>
      <td>washington reuters house speaker paul ryan fri...</td>
      <td>[washington, reuters, house, speaker, paul, ry...</td>
      <td>washington reuters house speaker paul ryan fri...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>beijing reuters chinese president xi jinping i...</td>
      <td>0</td>
      <td>beijing reuters chinese president xi jinping s...</td>
      <td>[beijing, reuters, chinese, president, xi, jin...</td>
      <td>beijing reuters chinese president xi jinping s...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>paris reuters french president emmanuel macron...</td>
      <td>0</td>
      <td>paris reuters french president emmanuel macron...</td>
      <td>[paris, reuters, french, president, emmanuel, ...</td>
      <td>paris reuters french president emmanuel macron...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>moscow reuters russian general killed in syria...</td>
      <td>0</td>
      <td>moscow reuters russian general killed syria se...</td>
      <td>[moscow, reuters, russian, general, killed, sy...</td>
      <td>moscow reuters russian general kill syria seco...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>washington reuters president donald trump prop...</td>
      <td>0</td>
      <td>washington reuters president donald trump prop...</td>
      <td>[washington, reuters, president, donald, trump...</td>
      <td>washington reuters president donald trump prop...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>shanghai reuters police in beijing have detain...</td>
      <td>0</td>
      <td>shanghai reuters police beijing detained eight...</td>
      <td>[shanghai, reuters, police, beijing, detained,...</td>
      <td>shanghai reuters police beijing detain eightee...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>london reuters nigel farage an anti immigratio...</td>
      <td>0</td>
      <td>london reuters nigel farage anti immigration p...</td>
      <td>[london, reuters, nigel, farage, anti, immigra...</td>
      <td>london reuters nigel farage anti immigration p...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_v4 = df_v3.copy()  # Create a copy of the original DataFrame
df_v4 =  df_v4.drop(['text', 'text_without_stopwords', 'tokenized_text'], axis=1)  # Drop the original 'text', 'text_without_stopwords', and 'tokenized_text' columns
df_v4.reset_index(drop=True, inplace=True)  # Reset the index of the DataFrame
df_v4.to_csv('processed_data4.csv', index=False)
```


```python
df_v4.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>lemmatized_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>donald trump administration seem austerity lon...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>sydney reuters watergate scandal pale comparis...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>follow covfefe gate trump make absolute fool s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>reuters wife treasury secretary steve mnuchin ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>chicago reuters year long audit program overse...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>beijing reuters china learnt history allow mao...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>ankara reuters european union share turkey vie...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>washington reuters state department approve po...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>washington reuters house speaker paul ryan fri...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>beijing reuters chinese president xi jinping s...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>paris reuters french president emmanuel macron...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>moscow reuters russian general kill syria seco...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>washington reuters president donald trump prop...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>shanghai reuters police beijing detain eightee...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>london reuters nigel farage anti immigration p...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from wordcloud import WordCloud

# wordcloud for real news
consolidated = ' '.join( 
    word for word in df_v4['lemmatized_text'][df_v4['label'] == 0].astype(str)) 
wordCloud = WordCloud(width=1600, 
                      height=800, 
                      random_state=21, 
                      max_font_size=110, 
                      collocations=False) 
plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.text(0.5, 0.98, 'Word Cloud of Lemmatized Text (Real News)\n', ha='center', fontsize=20, transform=plt.gca().transAxes)
plt.axis('off') 
plt.show() 
```


    
![png](https://i.imgur.com/GOCYtAZ.png)
    



```python
# wordcloud for fake news
consolidated = ' '.join( 
    word for word in df_v4['lemmatized_text'][df_v4['label'] == 1].astype(str)) 
wordCloud = WordCloud(width=1600, 
                      height=800, 
                      random_state=21, 
                      max_font_size=110, 
                      collocations=False) 
plt.figure(figsize=(15, 10)) 
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
plt.text(0.5, 0.98, 'Word Cloud of Lemmatized Text (Fake News)\n', ha='center', fontsize=20, transform=plt.gca().transAxes)
plt.axis('off')
plt.show() 
```


    
![png](https://i.imgur.com/e2y0PRA.png)
    



```python
from sklearn.feature_extraction.text import CountVectorizer

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df_v4['lemmatized_text'], 20)
df_temp = pd.DataFrame(common_words, columns=['Review', 'count'])

df_temp.groupby('Review').sum()['count'].sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 6),
    xlabel="Top Words",
    ylabel="Count",
    title="Bar Chart of Top Words Frequency"
)
plt.show()
```


    
![png](https://i.imgur.com/b85ET61.png)
    


##### 4.5 Splitting the dataset into training and testing sets


```python
from sklearn.model_selection import train_test_split

X = df_v4['lemmatized_text']
y = df_v4['label']  # Target variable

#  split the dataset into training and testing sets 75% for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12, stratify=y)


#vectorization (tf-idf)
from sklearn.feature_extraction.text import TfidfVectorizer

# apply TF-IDF for text column
tfidf_vectorizer = TfidfVectorizer()
XV_train = tfidf_vectorizer.fit_transform( X_train)
XV_test = tfidf_vectorizer.transform( X_test)
```


```python
print("The shape of the train set: ",XV_train.shape)
print("The shape of the test set: ",XV_test.shape)
```

    The shape of the train set:  (27126, 78730)
    The shape of the test set:  (9042, 78730)
    


```python
# print the first 20 rows of the training set
print(XV_train[:20])
```

      (0, 18801)	0.02907515634524964
      (0, 8955)	0.0346416891962956
      (0, 23038)	0.0213022704660247
      (0, 23265)	0.01900714659085048
      (0, 64833)	0.0278220695552509
      (0, 12486)	0.02383578302317496
      (0, 57802)	0.028749648006821643
      (0, 20529)	0.01883926037295328
      (0, 50655)	0.03179328374340753
      (0, 39279)	0.04044597656188198
      (0, 66592)	0.019791741095308124
      (0, 13454)	0.017911738002520085
      (0, 15265)	0.04678466080409881
      (0, 71951)	0.022053861156716273
      (0, 31499)	0.03684487704676029
      (0, 4836)	0.025482207449826048
      (0, 56072)	0.023137880881111118
      (0, 60200)	0.12664073018300193
      (0, 76571)	0.03376201682466219
      (0, 17448)	0.025409770781884444
      (0, 52283)	0.042765294802600294
      (0, 39845)	0.03501017780837866
      (0, 24190)	0.03618956068273138
      (0, 33763)	0.020074717990148105
      (0, 21157)	0.03813266714104718
      :	:
      (19, 72098)	0.01221134693552258
      (19, 56944)	0.02950266905598295
      (19, 51429)	0.02130078089495209
      (19, 7861)	0.03525878225196407
      (19, 27203)	0.0510681307769709
      (19, 2550)	0.02691609528632083
      (19, 62287)	0.02846247700167592
      (19, 45795)	0.02404603540908287
      (19, 41666)	0.015269660747135318
      (19, 14884)	0.03007891356424561
      (19, 62584)	0.026243289088044658
      (19, 16386)	0.01982832576713707
      (19, 55406)	0.036764016161186845
      (19, 74612)	0.034063290417496235
      (19, 68273)	0.03204251340979707
      (19, 53617)	0.031017508303428316
      (19, 35692)	0.07600391645880775
      (19, 68972)	0.01664452446296226
      (19, 27857)	0.0268928116195057
      (19, 53548)	0.021868529337252446
      (19, 13374)	0.05554922817566135
      (19, 47726)	0.0243244389075812
      (19, 14756)	0.03604875415226504
      (19, 61186)	0.009930911752022386
      (19, 20749)	0.020330923605580295
    

## 5 The Model

##### 5.1 Noramlization of the data


```python
from sklearn.preprocessing import Normalizer

# Initialize Normalizer
normalizer = Normalizer()

# Normalize the training set
XV_train_normalized = normalizer.fit_transform(XV_train)

# Normalize the test set
XV_test_normalized = normalizer.transform(XV_test) 
```


```python

print(XV_train_normalized[:20])
```

      (0, 18801)	0.029075156345249632
      (0, 8955)	0.034641689196295594
      (0, 23038)	0.021302270466024696
      (0, 23265)	0.019007146590850475
      (0, 64833)	0.027822069555250892
      (0, 12486)	0.023835783023174954
      (0, 57802)	0.028749648006821636
      (0, 20529)	0.018839260372953276
      (0, 50655)	0.03179328374340752
      (0, 39279)	0.04044597656188197
      (0, 66592)	0.01979174109530812
      (0, 13454)	0.01791173800252008
      (0, 15265)	0.0467846608040988
      (0, 71951)	0.02205386115671627
      (0, 31499)	0.03684487704676028
      (0, 4836)	0.02548220744982604
      (0, 56072)	0.023137880881111114
      (0, 60200)	0.1266407301830019
      (0, 76571)	0.033762016824662185
      (0, 17448)	0.025409770781884437
      (0, 52283)	0.04276529480260029
      (0, 39845)	0.03501017780837865
      (0, 24190)	0.036189560682731374
      (0, 33763)	0.0200747179901481
      (0, 21157)	0.038132667141047176
      :	:
      (19, 72098)	0.012211346935522576
      (19, 56944)	0.029502669055982943
      (19, 51429)	0.021300780894952086
      (19, 7861)	0.03525878225196406
      (19, 27203)	0.05106813077697089
      (19, 2550)	0.026916095286320822
      (19, 62287)	0.028462477001675913
      (19, 45795)	0.02404603540908286
      (19, 41666)	0.015269660747135315
      (19, 14884)	0.030078913564245604
      (19, 62584)	0.02624328908804465
      (19, 16386)	0.019828325767137066
      (19, 55406)	0.03676401616118684
      (19, 74612)	0.03406329041749623
      (19, 68273)	0.032042513409797066
      (19, 53617)	0.03101750830342831
      (19, 35692)	0.07600391645880773
      (19, 68972)	0.016644524462962256
      (19, 27857)	0.026892811619505692
      (19, 53548)	0.021868529337252442
      (19, 13374)	0.05554922817566134
      (19, 47726)	0.024324438907581195
      (19, 14756)	0.036048754152265036
      (19, 61186)	0.009930911752022384
      (19, 20749)	0.02033092360558029
    

##### 5.2 Building the Model 1

##### 5.2.1  Logistic Regression


```python
from sklearn.linear_model import LogisticRegression       # to apply the Logistic regression

#  create a logistic regression model and fit it to the training data
model = LogisticRegression();                          # create a logistic regression model
model.fit(XV_train_normalized, y_train)                      # fit the model to the training data
yhat = model.predict(XV_test)                    # make predictions on the test data
```

##### 5.2.2 Model Evaluation


```python
train_accuracy = model.score(XV_train_normalized, y_train)   # calculate the training accuracy
print(f'The accuracy for the training set is {100 * train_accuracy:.2f}%')  # print the training accuracy
test_accuracy = model.score(XV_test_normalized, y_test)                          # calculate the test accuracy
print(f'The accuracy for the test set is {100 * test_accuracy:.2f}%')       # print the test accuracy
```

    The accuracy for the training set is 98.94%
    The accuracy for the test set is 98.20%
    

##### 5.2.3 Confusion Matrix and report


```python
from sklearn import metrics # used to compute accuracy

#  compute the confusion matrix and classification report for the test set
print(metrics.confusion_matrix(y_test, yhat))
```

    [[5240   53]
     [ 110 3639]]
    


```python
TP = np.sum(np.logical_and(yhat == 0, y_test == 0)) # true positives
TN = np.sum(np.logical_and(yhat == 1, y_test == 1))  # true negatives
FP = np.sum(np.logical_and(yhat == 1, y_test == 0))  # false positives
FN = np.sum(np.logical_and(yhat == 0, y_test == 1)) # false negatives

print(f'TP: {TP:4} , FP: {FP:4}')
print(f'FN: {FN:4} , TN: {TN:4}')
```

    TP: 5240 , FP:   53
    FN:  110 , TN: 3639
    


```python
# print the classification report for the test set
print(metrics.classification_report(y_test, yhat))
```

                  precision    recall  f1-score   support
    
               0       0.98      0.99      0.98      5293
               1       0.99      0.97      0.98      3749
    
        accuracy                           0.98      9042
       macro avg       0.98      0.98      0.98      9042
    weighted avg       0.98      0.98      0.98      9042
    
    


```python
# Confusion matrix of Results from Decision Tree classification 
cm = metrics.confusion_matrix(y_test, yhat) 
  
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, 
                                            display_labels=[False, True]) 
  
cm_display.plot() 
plt.show() 
```


    
![png](https://i.imgur.com/py60dL5.png)
    


##### 5.3 Building the Model 2


```python
# Determine time_steps and input_features
time_steps = XV_train_normalized.shape[0]  # Length of TF-IDF vectors
input_features = XV_train_normalized.shape[1]  # Number of features in TF-IDF vectors
print(f'The length of the TF-IDF vectors is {time_steps} and the number of features is {input_features}.')

```

    The length of the TF-IDF vectors is 27126 and the number of features is 78730.
    


```python
#!pip install tensorflow
```


```python
X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(df_v4['lemmatized_text'], df_v4['label'], test_size=0.25, random_state=18)
```


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_vocab = 10000 # maximum number of words in the vocabulary
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train_rnn) # fit the tokenizer on the text data
```


```python
X_train_rnn = tokenizer.texts_to_sequences(X_train_rnn)
X_test_rnn = tokenizer.texts_to_sequences(X_test_rnn)
```


```python
X_train_rnn = pad_sequences(X_train_rnn, padding='post', maxlen=256)
X_test_rnn = pad_sequences(X_test_rnn, padding='post', maxlen=256)
```


```python
import tensorflow as tf
import keras
import pydot
import graphviz
from keras import layers
from keras.utils.vis_utils import plot_model


model_rnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
```


```python
import time

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model_rnn.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['accuracy', keras.metrics.Recall()])

start_time = time.time()

history = model_rnn.fit(X_train_rnn, y_train_rnn, epochs=4,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])

end_time = time.time()

# Calculate and print the duration
duration = end_time - start_time
print(f"Training took {duration:.2f} seconds.")
```

    Epoch 1/4
    814/814 [==============================] - 264s 316ms/step - loss: 0.0938 - accuracy: 0.9564 - recall: 0.9077 - val_loss: 0.0331 - val_accuracy: 0.9900 - val_recall: 0.9810
    Epoch 2/4
    814/814 [==============================] - 226s 277ms/step - loss: 0.0185 - accuracy: 0.9945 - recall: 0.9910 - val_loss: 0.0384 - val_accuracy: 0.9893 - val_recall: 0.9909
    Epoch 3/4
    814/814 [==============================] - 230s 283ms/step - loss: 0.0063 - accuracy: 0.9982 - recall: 0.9973 - val_loss: 0.0387 - val_accuracy: 0.9904 - val_recall: 0.9946
    Training took 720.20 seconds.
    


```python
model_rnn.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 128)         1280000   
                                                                     
     bidirectional (Bidirectiona  (None, None, 128)        98816     
     l)                                                              
                                                                     
     bidirectional_1 (Bidirectio  (None, 32)               18560     
     nal)                                                            
                                                                     
     dense (Dense)               (None, 64)                2112      
                                                                     
     dropout (Dropout)           (None, 64)                0         
                                                                     
     dense_1 (Dense)             (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 1,399,553
    Trainable params: 1,399,553
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# Confusion matrix of Results from Decision Tree classification 

yhat_rnn = model_rnn.predict(X_test_rnn)
yhat_rnn = (yhat_rnn > 0.5)
yhat_rnn = yhat_rnn.astype(int)

cm_rnn = metrics.confusion_matrix(y_test_rnn, yhat_rnn) 
  
cm_display_rnn = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_rnn, 
                                            display_labels=[False, True]) 
  
cm_display_rnn.plot() 
plt.show() 
```

    283/283 [==============================] - 23s 74ms/step
    


    
![png](https://i.imgur.com/HrODimw.png)
    



```python
def plotgraphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history[string])
  plt.xlabel('Epochs')
  plt.xticks(ticks=history.epoch)
  plt.ylabel(string)
  plt.legend([string])
  plt.title(string.upper() + ' vs. Epochs')
  plt.show()
```


```python
plotgraphs(history, 'accuracy')
```


    
![png](https://i.imgur.com/93PGWB9.png)
    



```python
plotgraphs(history, 'loss')
```


    
![png](https://i.imgur.com/d4ZWevD.png)
    



```python

```
