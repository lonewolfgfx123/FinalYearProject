import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics


# Read dataset to pandas dataframe
dataset = pd.read_csv('DataSet.csv')  
dataset.head()


LabelEncoder = preprocessing.LabelEncoder()
for comName in dataset.columns.values:
    dataset[comName] = LabelEncoder.fit_transform(dataset[comName])
    

X = dataset.iloc[:, [3]].values  
y = dataset.iloc[:, 4].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 4)
 


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print( metrics.accuracy_score(y_test,y_pred))
