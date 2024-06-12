import numpy as np
import pandas as pd
import lightgbm
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,log_loss,precision_recall_curve,average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from collections import Counter
from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size': 18})
#plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
#plt.style.use('seaborn-paper')
import copy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


models = {
        'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbours': KNeighborsClassifier(),
        'Support Vector Machine': SVC(),
        'Naive Bayes': BernoulliNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=5,min_samples_split=2),
        'Random Forest': RandomForestClassifier(n_estimators=100,min_samples_split=2),
        'LightGBM':LGBMClassifier(max_depth=7,
                                  boosting_type='gbdt',
                                  objective='binary',
                                  learning_rate=0.05,
                                  n_estimators=100,
                                  num_leaves=2 ** 5-1
                                 ),
        'GBDT':XGBClassifier(max_depth=7,
                                learning_rate=0.1,
                                n_estimators=150)
}
#
if __name__=="__main__":
    df_317 = pd.read_csv('data_processed/df_317.csv')
    g_frts=[]
    for k in df_317.columns:
        if k.startswith('g__'):
            g_frts.append(k)
    #
    select_frts=g_frts[::]
    print('len(select_frts):', len(select_frts))
    #
    # train_x = df_317[~ ((df_317['cancer']==1) & (df_317['MPN']==0))][select_frts].copy()
    # train_y = df_317[~ ((df_317['cancer']==1) & (df_317['MPN']==0))]["MPN"]
    train_x = df_317[select_frts].copy()
    train_y = df_317["MPN"]
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_x)
    print(train_x.shape,train_y.shape)
    # train_x = train_x.values
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024) 
    results = {}
    for model_name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, train_y, cv=kf, scoring='roc_auc')
        results[model_name] = scores.mean()
        #break
    print(results)