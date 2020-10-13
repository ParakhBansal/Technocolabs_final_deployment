import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\pb\Documents\technocolabs final\deployment\cleaned_data.csv')

items_to_remove = ['PAY_1','ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 
                   'others', 'university']

df = df.drop(items_to_remove,axis=1)

rf = RandomForestClassifier\
(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)

X = df.drop(['default payment next month'] ,axis=1)
y= df['default payment next month']

rf.fit(X.values,y.values)

pickle.dump(rf, open('rfmodel.pkl','wb'))