import pandas as pd
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



def linearregression(x_test):
    data = pd.read_csv('dataset.csv')
    data.head()
    
    X = data.iloc[:,[3]].values
    y = data.iloc[:,4].values
    
    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)
    
    
    return model.predict(x_test)

    
def KNN(x_test):
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
    return knn.predict(x_test)
    


dataset = pd.read_csv('DataSet.csv')  
dataset.head()
    
    
LabelEncoder = preprocessing.LabelEncoder()
for comName in dataset.columns.values:
    dataset[comName] = LabelEncoder.fit_transform(dataset[comName])
        
    
X = dataset.iloc[:, [3]].values  
y = dataset.iloc[:, 4].values
     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 4)
print("Average score:")
print((linearregression(X_test) + KNN(X_test)) / 2)
