I have been exploring Nearest Neighbours Methods which are commonly adopted for solving classification problems. In the context of Indoor localisation, such methods are employed to identify LoS and NLoS propagation modes. This blog post basically provides a Jupyter Notebook to try this classification on your own using openly available data-set for UWB Localisation.
The link for Jupyter notebook is [here]() and the Markdown version is provided in this post.

# Notebook Outlining the Usage of KNN for LoS/NLoS classification

This Notebook provides a brief introduction to usage of KNN base LoS/NLoS classification. We employ the UWB Localisation Dataset openly available at: http://log-a-tec.eu/uwb-ds.html
You should download and extract the dataset in a subfolder name data
You also need to remove first version line from all csv files

We utilise two parameters, i.e. RSS of First Path and Channel Impulse response amplitude of first path.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from IPython.display import display, HTML


path = r'data/office/measurements/1.07_9.37_1.2/'                     # use your path
all_files = glob.glob(path+"*.csv")     # advisable to use os.path.join as this makes concatenation OS independent
fields =['NLOS','FP_POINT1','RSS_FP']
df_from_each_file = (pd.read_csv(f,skipinitialspace=True, usecols=fields) for f in all_files)
df   = pd.concat(df_from_each_file, ignore_index=True)
display(df)
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
      <th>NLOS</th>
      <th>RSS_FP</th>
      <th>FP_POINT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>LOS</td>
      <td>-78.194451</td>
      <td>-80.535661</td>
    </tr>
    <tr>
      <td>1</td>
      <td>LOS</td>
      <td>-78.338668</td>
      <td>-80.627503</td>
    </tr>
    <tr>
      <td>2</td>
      <td>LOS</td>
      <td>-78.216480</td>
      <td>-81.031538</td>
    </tr>
    <tr>
      <td>3</td>
      <td>LOS</td>
      <td>-78.240909</td>
      <td>-81.179942</td>
    </tr>
    <tr>
      <td>4</td>
      <td>LOS</td>
      <td>-78.345567</td>
      <td>-81.243439</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1483</td>
      <td>NLOS</td>
      <td>-82.590764</td>
      <td>-86.272480</td>
    </tr>
    <tr>
      <td>1484</td>
      <td>NLOS</td>
      <td>-81.783098</td>
      <td>-85.618811</td>
    </tr>
    <tr>
      <td>1485</td>
      <td>NLOS</td>
      <td>-81.978536</td>
      <td>-86.146448</td>
    </tr>
    <tr>
      <td>1486</td>
      <td>NLOS</td>
      <td>-82.571613</td>
      <td>-86.140362</td>
    </tr>
    <tr>
      <td>1487</td>
      <td>NLOS</td>
      <td>-81.873701</td>
      <td>-85.752665</td>
    </tr>
  </tbody>
</table>
<p>1488 rows × 3 columns</p>
</div>


Filter the LoS/NLoS data for visualisation


```python
NLoSFilter = df[df['NLOS'] == 'NLOS'];
LoSFilter = df[df['NLOS'] == 'LOS'];
NLoSFilter.head()
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
      <th>NLOS</th>
      <th>RSS_FP</th>
      <th>FP_POINT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>31</td>
      <td>NLOS</td>
      <td>-79.317305</td>
      <td>-84.496205</td>
    </tr>
    <tr>
      <td>32</td>
      <td>NLOS</td>
      <td>-79.392652</td>
      <td>-85.282834</td>
    </tr>
    <tr>
      <td>33</td>
      <td>NLOS</td>
      <td>-79.266979</td>
      <td>-84.491246</td>
    </tr>
    <tr>
      <td>34</td>
      <td>NLOS</td>
      <td>-79.533815</td>
      <td>-84.698210</td>
    </tr>
    <tr>
      <td>35</td>
      <td>NLOS</td>
      <td>-79.811188</td>
      <td>-85.720904</td>
    </tr>
  </tbody>
</table>
</div>




```python
LoSFilter.head()
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
      <th>NLOS</th>
      <th>RSS_FP</th>
      <th>FP_POINT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>LOS</td>
      <td>-78.194451</td>
      <td>-80.535661</td>
    </tr>
    <tr>
      <td>1</td>
      <td>LOS</td>
      <td>-78.338668</td>
      <td>-80.627503</td>
    </tr>
    <tr>
      <td>2</td>
      <td>LOS</td>
      <td>-78.216480</td>
      <td>-81.031538</td>
    </tr>
    <tr>
      <td>3</td>
      <td>LOS</td>
      <td>-78.240909</td>
      <td>-81.179942</td>
    </tr>
    <tr>
      <td>4</td>
      <td>LOS</td>
      <td>-78.345567</td>
      <td>-81.243439</td>
    </tr>
  </tbody>
</table>
</div>



Visualisation of LoS/NLoS clusters


```python
plt.scatter(NLoSFilter['RSS_FP'],NLoSFilter['FP_POINT1'],c='red')
plt.scatter(LoSFilter['RSS_FP'],LoSFilter['FP_POINT1'],c='blue')
plt.xlabel('RSS_FP')
plt.ylabel('FP_POINT1')
plt.show()
```


![png](output_8_0.png)



```python
clabel = df['NLOS']
X = df[['RSS_FP','FP_POINT1']]
X
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
      <th>RSS_FP</th>
      <th>FP_POINT1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>-78.194451</td>
      <td>-80.535661</td>
    </tr>
    <tr>
      <td>1</td>
      <td>-78.338668</td>
      <td>-80.627503</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-78.216480</td>
      <td>-81.031538</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-78.240909</td>
      <td>-81.179942</td>
    </tr>
    <tr>
      <td>4</td>
      <td>-78.345567</td>
      <td>-81.243439</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1483</td>
      <td>-82.590764</td>
      <td>-86.272480</td>
    </tr>
    <tr>
      <td>1484</td>
      <td>-81.783098</td>
      <td>-85.618811</td>
    </tr>
    <tr>
      <td>1485</td>
      <td>-81.978536</td>
      <td>-86.146448</td>
    </tr>
    <tr>
      <td>1486</td>
      <td>-82.571613</td>
      <td>-86.140362</td>
    </tr>
    <tr>
      <td>1487</td>
      <td>-81.873701</td>
      <td>-85.752665</td>
    </tr>
  </tbody>
</table>
<p>1488 rows × 2 columns</p>
</div>



Code for Knn similar to Scikit learn blog and stack abuse:https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,clabel, test_size=0.20)
```


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```


```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')




```python
plt.scatter(X_train[:,0],X_train[:,1])
```




    <matplotlib.collections.PathCollection at 0x2386c9b1f88>




![png](output_14_1.png)



```python
y_pred = classifier.predict(X_test)
display(y_test==y_pred)
```


    1209     True
    987      True
    1048    False
    211      True
    1182     True
            ...  
    349      True
    1478     True
    132      True
    22       True
    728      True
    Name: NLOS, Length: 298, dtype: bool



```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    [[ 55  16]
     [ 20 207]]
                  precision    recall  f1-score   support
    
             LOS       0.73      0.77      0.75        71
            NLOS       0.93      0.91      0.92       227
    
        accuracy                           0.88       298
       macro avg       0.83      0.84      0.84       298
    weighted avg       0.88      0.88      0.88       298
    
    


```python

```
