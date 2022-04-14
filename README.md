## Regression

Models:
- RandomForestRegressor
- Ridge
- LinearRegression
- Lasso
- DecisionTreeRegressor
- SVR

Metrics:
- `sklearn.metrics.r2_score`
- `sklearn.metrics.mean_absolute_error`
- `sklearn.metrics.mean_squared_error`

## Classification

Models:
- RandomForestClassifier
- KNN
- LogisticRegression (binary clf)
- DecisionTreeClassifier
- LinearSVC

Metrics:
- accuracy
- precision
- recall
- f1
- `sns.heatmap` x confusion_matrix
- classification_report

## Tips & Tricks

### Encoding categorical features

```py
obj_list = data.select_dtypes(include='object').columns # get column that is 'object'
```

```py
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for obj in obj_list:
    data[obj] = le.fit_transform(data[obj].astype(str)) # encode each categorical column
```

### Fill NA values for a column

```py
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median') # fill each with the column's median value
imputer.fit(X[:, -2:])
X[:, -2:] = imputer.transform(X[:, -2:])
```

### Feature Scaling

Only applies to `X_train` and `X_test`.

```py
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train) # normalize
X_test = sc.fit_transform(X_test) # normalize
```

### Imbalanced classification

Use SMOTE

```py
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=10, random_state=42)
X, y = sm.fit_resample(X, y)
```

### Classification analysis

[https://github.com/dformoso/sklearn-classification](https://github.com/dformoso/sklearn-classification)