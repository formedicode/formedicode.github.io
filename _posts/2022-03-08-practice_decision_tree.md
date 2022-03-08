---
layout : single
title : 'Decision Tree model 실습예제'
```python
import pandas as pd

df = pd.DataFrame({'weather' : ['sunny','sunny', 'overcast', 'rainy', 'rainy', 
                                'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 
                                'sunny', 'overcast', 'overcast','rainy'], 
                   'hum' : ['high', 'high', 'high', 'high', 'normal', 
                           'normal','normal', 'high', 'normal', 'normal', 
                           'normal', 'high', 'normal', 'high'], 
                   'wind' : ['weak', 'strong', 'weak', 'weak', 'weak', 
                             'strong', 'strong', 'weak', 'weak', 'weak', 
                             'strong', 'strong', 'weak', 'strong'], 
                   'play': ['no','no','yes', 'yes', 'yes', 
                            'no', 'yes', 'no', 'yes','yes',
                             'yes', 'yes', 'yes', 'no'] 
                  })
```


```python
df
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
      <th>weather</th>
      <th>hum</th>
      <th>wind</th>
      <th>play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>sunny</td>
      <td>high</td>
      <td>weak</td>
      <td>no</td>
    </tr>
    <tr>
      <td>1</td>
      <td>sunny</td>
      <td>high</td>
      <td>strong</td>
      <td>no</td>
    </tr>
    <tr>
      <td>2</td>
      <td>overcast</td>
      <td>high</td>
      <td>weak</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>3</td>
      <td>rainy</td>
      <td>high</td>
      <td>weak</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>4</td>
      <td>rainy</td>
      <td>normal</td>
      <td>weak</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>5</td>
      <td>rainy</td>
      <td>normal</td>
      <td>strong</td>
      <td>no</td>
    </tr>
    <tr>
      <td>6</td>
      <td>overcast</td>
      <td>normal</td>
      <td>strong</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>7</td>
      <td>sunny</td>
      <td>high</td>
      <td>weak</td>
      <td>no</td>
    </tr>
    <tr>
      <td>8</td>
      <td>sunny</td>
      <td>normal</td>
      <td>weak</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>9</td>
      <td>rainy</td>
      <td>normal</td>
      <td>weak</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>10</td>
      <td>sunny</td>
      <td>normal</td>
      <td>strong</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>11</td>
      <td>overcast</td>
      <td>high</td>
      <td>strong</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>12</td>
      <td>overcast</td>
      <td>normal</td>
      <td>weak</td>
      <td>yes</td>
    </tr>
    <tr>
      <td>13</td>
      <td>rainy</td>
      <td>high</td>
      <td>strong</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
# step1. 숫자(빈도값으로 변경)
from sklearn import preprocessing     # 사이킷런 - preprocessing 모듈 
```


```python
le = preprocessing.LabelEncoder()     #labelencode method 사용(명목형 -> 숫자형) 
```


```python
def convert_encoder(df, col):
    label_list  = le.fit_transform(df[col])
    return label_list
```


```python
weather_encoder = convert_encoder(df, 'weather')
hum_encoder  = convert_encoder(df, 'hum')
wind_encoder = convert_encoder(df, 'wind')
label = convert_encoder(df, 'play')
```


```python
display(weather_encoder)
display(hum_encoder)
display(wind_encoder)
display(label)
```


    array([2, 2, 0, 1, 1, 1, 0, 2, 2, 1, 2, 0, 0, 1])



    array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0])



    array([1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0])



    array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])



```python
features = list(zip(weather_encoder, hum_encoder, wind_encoder)) #zip은 길이가 같을때 사용, 튜플형태로 반환
display(features)
```


    [(2, 0, 1),
     (2, 0, 0),
     (0, 0, 1),
     (1, 0, 1),
     (1, 1, 1),
     (1, 1, 0),
     (0, 1, 0),
     (2, 0, 1),
     (2, 1, 1),
     (1, 1, 1),
     (2, 1, 0),
     (0, 0, 0),
     (0, 1, 1),
     (1, 0, 0)]


 ## Model1 : DecisionTreeClassifier()


```python
#모듈 
from sklearn.tree import DecisionTreeClassifier

#모델선언
dt = DecisionTreeClassifier()

#모델학습 
dt.fit(features, label)

#모델예측 
pred = dt.predict(features)
```


```python
print(pred)
print(label)
print((pred == label).mean())
```

    [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    1.0


# 예제1 - Iris


```python
#모듈
from sklearn import datasets

#데이터로딩 
iris = datasets.load_iris()

#데이터설명 
print(iris.DESCR)
```

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...



```python
#데이터셋 불러오기 
x_data = iris.data
y_data = iris.target 
```


```python
#데이터셋 나누기 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                    test_size = 0.3, 
                                                    stratify = y_data, 
                                                    random_state = 777)

print(x_train.shape, y_train.shape)
```

    (105, 4) (105,)



```python
#모듈불러오기 sklearn의 tree
from sklearn.tree import DecisionTreeClassifier

#모델선언하기
dt = DecisionTreeClassifier()

#모델학습하기 
dt.fit(x_train, y_train)

#모델예측 
pred = dt.predict(x_test)
```


```python
#모델성능평가하기 
from sklearn import metrics

#정확도
print(f'DecisionTree accuracy score : {metrics.accuracy_score(y_test, pred)}\n') #실제 답지(y_test)-예측한 값(pred)와 93% 맞춤

#confusion matrix 
print(f'DecisionTree confusion metrics : \n{metrics.confusion_matrix(y_test, pred)}') #class별로 어떻게 분류했는지 볼수 있는 것 ()

```

    DecisionTree accuracy score : 0.9333333333333333
    
    DecisionTree confusion metrics : 
    [[15  0  0]
     [ 0 13  2]
     [ 0  1 14]]



```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cm2 = pd.DataFrame(metrics.confusion_matrix(y_test, pred))
sns.heatmap(cm2, annot=True, fmt = 'd', cmap = 'Reds')
plt.xlabel('class')
plt.ylabel('predict')
plt.show()
```


    <Figure size 640x480 with 2 Axes>


# 1-1 max_depth에 따른 DecisionTree model (overfitting 확인)


```python
for dep in range(2, 10):
    
    #모델선언
    dt = DecisionTreeClassifier(max_depth = dep, random_state = 777) # max_depth = depth(최대 해당 depth까지만 진행) 
                                                                     # random_state(고정값을 두어, depth가 변동할때 정확도 평가)
    #모델학습
    dt.fit(x_train, y_train)  
    
    #모델예측
    pred = dt.predict(x_test)
    
    #성능출력(정확도)
    print("DecisionTree depth",dep, ":", metrics.accuracy_score(y_test, pred)) # metrics.accuracy_score(y_test, pred)
```

    DecisionTree depth 2 : 0.9333333333333333
    DecisionTree depth 3 : 0.9555555555555556
    DecisionTree depth 4 : 0.9333333333333333
    DecisionTree depth 5 : 0.9333333333333333
    DecisionTree depth 6 : 0.9333333333333333
    DecisionTree depth 7 : 0.9333333333333333
    DecisionTree depth 8 : 0.9333333333333333
    DecisionTree depth 9 : 0.9333333333333333


# 1-2 DecisionTree model의 feature_importances 속성값
    ## 중용한 속성들을 파악하여 -> 속성의 갯수를 줄이는(차원축소) 하는데 사용하기도함


```python
#모델선언
dt = DecisionTreeClassifier(max_depth = 3, random_state = 777) #criterion = geni index가 함!
                                                                    
#모델학습
dt.fit(x_train, y_train)  
    
#모델예측
pred = dt.predict(x_test)
    
#성능출력(정확도)
print(dep, ":", metrics.accuracy_score(y_test, pred)) 

#변수중요도 
dt.feature_importances_   #deicision tree의 feature importances : vales(0 = 사용되지 x) , 모든 속성들을 나누어 평가할수 없기 떄문에 랜덤하게 속성을 뽑아서 구현한 한계점 
```

    9 : 0.9555555555555556





    array([0.        , 0.        , 0.42152292, 0.57847708])



# 1-3 criterion = geni(default값), entrophy(변경값)


```python
#모델선언
dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 777) #위와 동일 조건, but measure함수(entrophy만 변경)
                                                                    
#모델학습
dt.fit(x_train, y_train)  
    
#모델예측
pred = dt.predict(x_test)
    
#성능출력(정확도)
print(dep, ":", metrics.accuracy_score(y_test, pred)) 

#변수중요도 
dt.feature_importances_   #deicision tree의 feature importances : vales(0 = 사용되지 x) , 모든 속성들을 나누어 평가할수 없기 떄문에 랜덤하게 속성을 뽑아서 구현한 한계점 
```

    9 : 0.9555555555555556





    array([0.        , 0.        , 0.32441532, 0.67558468])



# 1-4 시각화 (4차원의 속성 -> 2차원 좌표평면(차원축소) 분류관점에서) 


```python
import matplotlib.pyplot as plt 

plt.scatter(x_test[:, 2], x_test[:, 3], c = y_test) # x_test[:, 2] ==  x_test의 모든 행, 세번쨰 속성 (feacture importance)
                                                    # x_test[:, 3] ==  x_test의 모든 행, 네번째 속성 (feacture importance)
                                                    # c == color y_test의 라벨(3가지) 

plt.show()  
```


![png](output_25_0.png)


# 2. 앙상블-RandomForest


```python
#앙상블 - 랜덤포레스트 모듈
from sklearn.ensemble import RandomForestClassifier

#렌덤포레스트 모델 선언 
rf = RandomForestClassifier(n_estimators = 50, random_state = 777) #Randomforest는 생성할 트리갯수(n_estimators = n개),  

#랜덤포레스트 모델 학습 
rf.fit(x_train, y_train)

#랜덤포레스트 예측 
pred_rf = rf.predict(x_test)

```


```python
#성능평가 모듈 불러오기 (metrics)
from sklearn import metrics

print(f'RandomForest accuracy : {metrics.accuracy_score(y_test, pred_rf)}\n')
print(f'RandomForest Confuision Matrix : \n{metrics.confusion_matrix(y_test, pred_rf)}') #대각선이 잘 맞춘것!
```

    RandomForest accuracy : 0.9333333333333333
    
    RandomForest Confuision Matrix : 
    [[15  0  0]
     [ 0 13  2]
     [ 0  1 14]]


## 결론:  보통은 DecisionTree < Randomforest 모델의 성능이 더 잘 나옴!

# 3. Boosting-Adaboost


```python
#Adaboost 모듈불러오기 
from sklearn.ensemble import AdaBoostClassifier

#AdaBoost 모델 선언 
adb = AdaBoostClassifier(n_estimators = 50, random_state = 777) #n_estimators = n (몇 번 반복할것인가) 

#AdaBoost 모델 학습
adb.fit(x_train, y_train)

#AdaBoost 모델 예측 
pred_adb = adb.predict(x_test)
```


```python
#성능평가 모듈 불러오기 (metrics)
from sklearn import metrics

print(f'Adaboost accuracy : {metrics.accuracy_score(y_test, pred_adb)}\n')
print(f'Adaboost Confuision Matrix : \n{metrics.confusion_matrix(y_test, pred_adb)}') #대각선이 잘 맞춘것!
```

    Adaboost accuracy : 0.9555555555555556
    
    Adaboost Confuision Matrix : 
    [[15  0  0]
     [ 0 14  1]
     [ 0  1 14]]


    ## 성능평가 : DecisionTree(0.93), max_depth = 3 (0.95)/Randomforest(0.93)/Adaboost(0.96) 
    ## 여러가지 모델을 사용해보고, 모델 내의 hyperparameter를 수정해서 사용할것!

# 예제 2 - Wine (iris 보다 더 큰 데이터셋) 


```python
#wine 데이터 로딩하기 
from sklearn import datasets
wine = datasets.load_wine()
print(wine.DESCR)
```

    .. _wine_dataset:
    
    Wine recognition dataset
    ------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 178 (50 in each of three classes)
        :Number of Attributes: 13 numeric, predictive attributes and the class
        :Attribute Information:
     		- Alcohol
     		- Malic acid
     		- Ash
    		- Alcalinity of ash  
     		- Magnesium
    		- Total phenols
     		- Flavanoids
     		- Nonflavanoid phenols
     		- Proanthocyanins
    		- Color intensity
     		- Hue
     		- OD280/OD315 of diluted wines
     		- Proline
    
        - class:
                - class_0
                - class_1
                - class_2
    		
        :Summary Statistics:
        
        ============================= ==== ===== ======= =====
                                       Min   Max   Mean     SD
        ============================= ==== ===== ======= =====
        Alcohol:                      11.0  14.8    13.0   0.8
        Malic Acid:                   0.74  5.80    2.34  1.12
        Ash:                          1.36  3.23    2.36  0.27
        Alcalinity of Ash:            10.6  30.0    19.5   3.3
        Magnesium:                    70.0 162.0    99.7  14.3
        Total Phenols:                0.98  3.88    2.29  0.63
        Flavanoids:                   0.34  5.08    2.03  1.00
        Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
        Proanthocyanins:              0.41  3.58    1.59  0.57
        Colour Intensity:              1.3  13.0     5.1   2.3
        Hue:                          0.48  1.71    0.96  0.23
        OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
        Proline:                       278  1680     746   315
        ============================= ==== ===== ======= =====
    
        :Missing Attribute Values: None
        :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML Wine recognition datasets.
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    
    The data is the results of a chemical analysis of wines grown in the same
    region in Italy by three different cultivators. There are thirteen different
    measurements taken for different constituents found in the three types of
    wine.
    
    Original Owners: 
    
    Forina, M. et al, PARVUS - 
    An Extendible Package for Data Exploration, Classification and Correlation. 
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.
    
    Citation:
    
    Lichman, M. (2013). UCI Machine Learning Repository
    [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science. 
    
    .. topic:: References
    
      (1) S. Aeberhard, D. Coomans and O. de Vel, 
      Comparison of Classifiers in High Dimensional Settings, 
      Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Technometrics). 
    
      The data was used with many others for comparing various 
      classifiers. The classes are separable, though only RDA 
      has achieved 100% correct classification. 
      (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) 
      (All results using the leave-one-out technique) 
    
      (2) S. Aeberhard, D. Coomans and O. de Vel, 
      "THE CLASSIFICATION PERFORMANCE OF RDA" 
      Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of 
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Journal of Chemometrics).
    



```python
#wine 데이터셋 나누기!
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, 
                                                    test_size = 0.3, 
                                                    stratify = wine.target, 
                                                    random_state = 777)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

    (124, 13) (54, 13) (124,) (54,)


# Deicision Tree Model - wine data


```python
#모듈 불러오기  
from sklearn.tree import DecisionTreeClassifier

#모델 선언
dct_wine = DecisionTreeClassifier(random_state = 7777)

#모델 학습하기 
dct_wine.fit(x_train, y_train)

#모델 예측하기 
pred_dct_wine = dct_wine.predict(x_test) 

#성능평가하기 
from sklearn import metrics

print(f'Decision Tree accuracy(wine_data): {metrics.accuracy_score(y_test, pred_dct_wine)}\n')
print(f'Decision Tree Confusion Matrics(wine_data): \n{metrics.confusion_matrix(y_test, pred_dct_wine)}')
```

    Decision Tree accuracy(wine_data): 0.9259259259259259
    
    Decision Tree Confusion Matrics(wine_data): 
    [[15  3  0]
     [ 0 20  1]
     [ 0  0 15]]


# Decision Tree Model(criterion 분류함수를 entrophy로 변경) - wine data


```python
#모듈 불러오기  
from sklearn.tree import DecisionTreeClassifier

#모델 선언
dct_wine = DecisionTreeClassifier(criterion = 'entropy',random_state = 7777)

#모델 학습하기 
dct_wine.fit(x_train, y_train)

#모델 예측하기 
pred_dct_wine = dct_wine.predict(x_test) 

#성능평가하기 
from sklearn import metrics

print(f'Decision Tree accuracy(wine_data): {metrics.accuracy_score(y_test, pred_dct_wine)}\n')
print(f'Decision Tree Confusion Matrics(wine_data): \n{metrics.confusion_matrix(y_test, pred_dct_wine)}')
```

    Decision Tree accuracy(wine_data): 0.8703703703703703
    
    Decision Tree Confusion Matrics(wine_data): 
    [[16  2  0]
     [ 0 19  2]
     [ 1  2 12]]


# 앙상블 모델(Random Forest) - wine data


```python
from sklearn.ensemble import RandomForestClassifier

rf_wine = RandomForestClassifier(n_estimators = 50, random_state = 7777) #Random forest는 몇개 트리 생서할지 (n_estimators 갯수 꼭 넣기!) 

rf_wine.fit(x_train, y_train)

pred_rf_wine = rf_wine.predict(x_test)

from sklearn import metrics

print(f'Random Forest accuracy(wine_data): {metrics.accuracy_score(y_test, pred_rf_wine)}\n')
print(f'Random Forest Confusion Matrics(wine_data): \n{metrics.confusion_matrix(y_test, pred_rf_wine)}')
```

    Random Forest accuracy(wine_data): 0.9814814814814815
    
    Random Forest Confusion Matrics(wine_data): 
    [[17  1  0]
     [ 0 21  0]
     [ 0  0 15]]



```python
from sklearn.ensemble import AdaBoostClassifier 

for step in range(2, 50):
    adb_wine = AdaBoostClassifier(n_estimators = step, random_state = 7777) #n_estimaor = 50 으로 고정했더니, 0.74로 overfitting 문제로 성능이 좋지 않음.

    adb_wine.fit(x_train, y_train)

    pred_adb_wine = adb_wine.predict(x_test)

    print(f'AdaBoost accuracy(wine_data) {step} : {metrics.accuracy_score(y_test, pred_adb_wine)}\n')
#     print(f'AdaBoost Confusion Matrics(wine_data): \n{metrics.confusion_matrix(y_test, pred_adb_wine)}')
```

    AdaBoost accuracy(wine_data) 2 : 0.7962962962962963
    
    AdaBoost accuracy(wine_data) 3 : 0.6666666666666666
    
    AdaBoost accuracy(wine_data) 4 : 0.8148148148148148
    
    AdaBoost accuracy(wine_data) 5 : 0.7037037037037037
    
    AdaBoost accuracy(wine_data) 6 : 0.8148148148148148
    
    AdaBoost accuracy(wine_data) 7 : 0.7407407407407407
    
    AdaBoost accuracy(wine_data) 8 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 9 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 10 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 11 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 12 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 13 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 14 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 15 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 16 : 0.7777777777777778
    
    AdaBoost accuracy(wine_data) 17 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 18 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 19 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 20 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 21 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 22 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 23 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 24 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 25 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 26 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 27 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 28 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 29 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 30 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 31 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 32 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 33 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 34 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 35 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 36 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 37 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 38 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 39 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 40 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 41 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 42 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 43 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 44 : 0.7407407407407407
    
    AdaBoost accuracy(wine_data) 45 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 46 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 47 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 48 : 0.7592592592592593
    
    AdaBoost accuracy(wine_data) 49 : 0.7407407407407407
    



```python
    ## 성능평가 : DecisionTree_geni index(0.93),DecisionTree_entrophy(0.87)/Randomforest(0.98)/Adaboost(0.66~0.81) 
    ## 여러가지 모델을 사용해보고, 모델 내의 hyperparameter를 수정해서 사용할것!
```


```python

```
