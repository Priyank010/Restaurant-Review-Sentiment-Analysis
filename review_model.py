

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = "\t", quoting = 3)


#Clean the data
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
     review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
     review = review.lower()
     review = review.split()
     ps = PorterStemmer()
     review = [ps.stem(words) for words in review if not words in set(stopwords.words('english'))]
     review = ' '.join(review)
     corpus.append(review)
     
#word of bags

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

pickle.dump(cv, open('cv-transform.pkl', 'wb'))

X_train,X_test,Y_train,Y_test = train_test_split(X,y)

rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,Y_train)


filename = 'restaurant-sentiment-mnb-model.pkl'
pickle.dump(rfc, open(filename, 'wb'))