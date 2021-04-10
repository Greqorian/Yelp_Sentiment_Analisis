# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
#  # Yelp Sentimental Analysis
#  ## Unternehmenssoftware | Data Science Project
# 
#  Project Group:
#  Gregor Pawlak email: s0563317@htw-berlin.de | Karsten Saß email: s0568771@htw-berlin.de
# 
#  ## Introduction
#  Project aim: Quick and clear classification of customers' feelings based on their written reviews for business.
# 
# 
#  ## Data Source
#  Yelp is company which is porviding the rating of the restaurants. It is internet based rating forum, where people can write and rate restaurants aroud the globe.
#  
#  Plese download the dataset business: yelp_academic_dataset_business.json and reviews: yelp_academic_dataset_review from  https://www.kaggle.com/yelp-dataset/yelp-dataset 

# %%
# package imports
#basics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#vizualisation
import matplotlib.pyplot as plt 
import seaborn as sns #Python data visualization library based on matplotlib
import multiprocessing

#miscellaneous
import gc #garbage processor
from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas() #progress bar optional


# my imports
import library as library

# scikit-learn machine learning
import sklearn.model_selection as model_selection
from nltk.corpus import stopwords #useless words/ stop words filtering for text processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# 0. The following program downloads datasets automatically from the kaggle server. By default it is inactive. You have the option to use it if the datasets are not in the origin folder.

# %%
# Download Datasets if not exists

reviewFile = './yelp_academic_dataset_review.json'
businessFile = './yelp_academic_dataset_business.json'
# library.downloadDataset(reviewFile, businessFile)

# %% [markdown]
#  # 1. Loading Datasets
# 
#  Loading JSON files. The load data are to big to be loaded at one, therefore we read the files line by line. Secondly, the each row is JSON but file is not JSON itself.
# 
#  Due to long loading time data reading is set for 100 000 first lines of the file. However the number was set for 1 000 000 lines for the presentation in Tableau.

# %%
help(library.loadJSON)


# %%

reviews = pd.DataFrame(library.loadJSON('./yelp_academic_dataset_review.json',{'review_id': [],'user_id': [], 'business_id': [],'stars': [], 'text':[], 'date':[]},size=100000))

business = pd.DataFrame(library.loadJSON('./yelp_academic_dataset_business.json',{'business_id': [], 'name':[], 'address':[], 'city':[], 'state': [], 'postal_code': [],'latitude': [], 'longitude': [], 'review_count': [], 'stars': [],'categories':[], },size=100000))

# helping the RAM
gc.collect()


# %%
reviews.head(3)


# %%
business.head(3)


# %%
reviews.info()

# %% [markdown]
#  # 2. Exploratory Data Analysis
#  The basic EDA is performed to find if there are extreme values and missing data.
# 
#  There are no missing or extreme values. The yelp dataset seems to be clear.

# %%
display(reviews.shape) #there are 1.000.000 dimensions/arrays with two elements stars and text
#number of missing values in the data set
display(reviews.isnull().sum(axis=0)) 

#basic statistical details like percentile, mean, std 
display(reviews.describe()) 

#spread map of missing values
sns.heatmap(reviews.isnull(), cbar=False) 

# there are no missing values shown

# %% [markdown]
#  # 3 Preprocess data
# 
#  Data preprocessing includes cleaning datasets from false values for correct data reading.
# 
#  ### 3.1 Cleaning regex

# %%
# utils_yelp.text_clean(x) will remove hypertext links, hashtags, @-symbol, numbers and empty spaces
# return list_of_words: list with words cleaned from the review.

reviews['text']=reviews['text'].progress_apply(lambda x: library.text_clean(x))

reviews.head(3)


# %%
reviews.info()

# %% [markdown]
#  ### 3.2 Cleaning punctuation

# %%
# Remove punctuation

reviews['text']=reviews['text'].progress_apply(lambda x: library.remove_punctuations(x))
reviews.head(5) 

# %% [markdown]
#  # 4. Data Modelling
# 
#  ### 4.1 Positive and Negative Labels
# 
#  In order to start training the model, we conduct a preliminary evaluation of user reviews. Positive reviews are those with ratings greater than 3.

# %%
# all stars >3 gets a true (1) and <3 false (0) in the column labels

reviews['labels'] = np.where(reviews['stars']>3,1,0)
reviews.head(10)


# %%
# The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model.

X_train_corp_t, X_test_corp_t, y_train_t, y_test_t = model_selection.train_test_split(reviews['text'], reviews['labels'],test_size=0.3, random_state=42)
display(X_train_corp_t.shape)
display(X_train_corp_t.shape)
display(y_train_t.shape)
display(y_test_t.shape)

# %% [markdown]
#  ### 4.2 Prepare vector with frequencies

# %%

# Transforms text to feature vectors that can be used as input to estimator. vocabulary_ Is a dictionary that converts each token (word) to feature index in the matrix, each unique token gets a feature index. In each vector the numbers (weights) represent features tf-idf score.

vector_Tfid =  TfidfVectorizer(min_df=100, ngram_range=(1, 1),stop_words='english')


X_train_t = vector_Tfid.fit(X_train_corp_t).transform(X_train_corp_t) 
X_test_t = vector_Tfid.transform(X_test_corp_t)
print(X_train_t.shape)

# %% [markdown]
#  ### 4.3 Model training

# %%
# Scikit-Learn. Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1

logreg = LogisticRegression(max_iter=500,solver='liblinear').fit(X_train_t,y_train_t)
   
print(f"LG accuracy trainnig set: {logreg.score(X_train_t, y_train_t)}")
print(f"LG accuracy test set: {logreg.score(X_test_t, y_test_t)}")

# %% [markdown]
# ### 4.4 Document Term Matrix
# 
# DTM describes the frequency of terms that occur in a collection of reviews. Allows to analyze the frequency of occurrence of terms for the whole set of data.

# %%
X = vector_Tfid.fit_transform(reviews['text']) #document_term_matrix or bag_of_words
dtm = pd.DataFrame(X.toarray())
dtm.columns = vector_Tfid.get_feature_names()

data_dtm = pd.concat([reviews, dtm], axis=1)

data_dtm.head(3)

# %% [markdown]
# # 5. Output
# ### 5.1 Most positive words

# %%

positiveWords = library.print_significant_words(logreg.coef_[0],1,15,vector_Tfid,True)
library.wc(positiveWords)

# %% [markdown]
#  ### 5.2 Most negative words

# %%
negativeWords = library.print_significant_words(logreg.coef_[0],0,15,vector_Tfid,True)
library.wc(negativeWords)

# %% [markdown]
#  # 6 Export Output with Split to Single Word
# 
#  ### 6.1 Merge negative and positive words
# 

# %%
# merge negative and positive words to the data frame

lists = [positiveWords, negativeWords]
frequent_words = pd.concat([pd.Series(x) for x in lists], axis=1)
frequent_words.columns = ['positives', 'negatives']


frequent_words

# %% [markdown]
#  ###  6.2 Inner Join Reviews and Business Tables

# %%
merged_inner = pd.merge(left=business, right=reviews, left_on='business_id', right_on='business_id')

merged_inner.head(3)

# %% [markdown]
#  ### 6.3 Save tables to xlsx format

# %%
merged_inner = merged_inner.rename(columns={'stars_x': 'stars_AVG', 'stars_y': 'stars'})


# %%
# reorganise table

merged_inner = merged_inner[['business_id',	'name',	'address','city','state','postal_code','latitude','longitude',	'review_count','stars_AVG','categories','review_id','user_id','stars','date','labels','text']]
merged_inner.head(3)


# %%
# Write Reviews Table to reviews.xlsx file

# reviews.to_excel(r'.\Reviews.xlsx')
# business.to_excel(r'.\Business.xlsx')

merged_inner.to_excel(r'.\Table_Merged.xlsx')


# %% [markdown]
# 6.3.1 Remove the # sign if you want to save frequent words to the excel file

# %%
# Write most frequent words to words.xlsx

#  frequent_words.to_excel(r'.\Words.xlsx')

# %% [markdown]
# ### 6.4 Split Reviews to single words   

# %%

table_splitted = merged_inner.join(merged_inner['text'].str.split(expand=True).add_prefix('text'))
del table_splitted['text']


# %%
table_reduced=table_splitted
# remove last 200 cloumns to speed up exporting to excel
# only 22 reviews exceed this length therefore no siginificant influence on the output
table_reduced.drop(table_reduced.iloc[:, 600:1015], inplace = True, axis = 1) 
#table_reduced.drop(table_reduced.columns[[-1,-20]], axis=1, inplace=True)

gc.collect()
table_reduced.head(3)


# %%
# Export to excel deactivated due to large memory amout needed to process

# table_reduced.to_excel(r'.\Table_splitted.xlsx')


# %%
table_reduced.to_csv(r'.\Table_splitted.csv')

# %% [markdown]
#  # 7. Export Output with Document Term Matrix
#  The export of data using the document term matrix will not be continued. The version with single word split has been selected.

# %%
# Merge Table version for Matrix
# merged_matrix = pd.merge(left=business, right=data_dtm, left_on='business_id', right_on='business_id')



# %%
#merged_matrix.to_excel(r'.\Merged_Matrix.xlsx')


# %%
#merged_matrix.head(3)

# %% [markdown]
#  # 8. Data Modelling | For Selected Place
# 
#  Sorting table to get the place with the highest count of reviews. Selected place is Bouchon Restaurant in Las Vegas

# %%
# Sort the table by number of reviews

merged_inner.sort_values(by=['review_count'], inplace=True, ascending=False)
merged_inner.head(3)


# %%
top_name = merged_inner['name'].iloc[0]
top_name


# %%
# select all entries for a business name "Bouchon"

is_name =  merged_inner['name']==top_name
selected_reviews = merged_inner[is_name]
print(selected_reviews.shape)


# %%
vector_Tfidf = TfidfVectorizer(min_df=100, ngram_range=(1, 1),stop_words='english')

# The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model.

X_train_corp_Tfidf, X_test_corp_Tfidf, y_train_Tfidf, y_test_Tfidf = model_selection.train_test_split(selected_reviews['text'], selected_reviews['labels'],test_size=0.3, random_state=42)

# Prepare vector with frequencies
# The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary


X_train_Tfidf = vector_Tfidf.fit(X_train_corp_Tfidf).transform(X_train_corp_Tfidf) 
X_test_Tfidf = vector_Tfidf.transform(X_test_corp_Tfidf)

# Scikit-Learn. Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1

logreg = LogisticRegression(max_iter=500,solver='liblinear').fit(X_train_Tfidf,y_train_Tfidf)


# %%
selected_positiveWords = library.print_significant_words(logreg.coef_[0],1,15,vector_Tfidf,True)
library.wc(selected_positiveWords)


# %%
selected_negativeWords = library.print_significant_words(logreg.coef_[0],0,15,vector_Tfidf,True)
library.wc(selected_negativeWords)

# %% [markdown]
# 

