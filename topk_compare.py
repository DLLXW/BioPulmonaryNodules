import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve
from lightgbm import LGBMClassifier
import os
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'Arial'
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
#
custom_colors = {
    'Mayo': 'tab:blue',
    'LightGBM': 'tab:cyan'
}
auc_figsize=(12,10)
ds = '20240520'
df_317 = pd.read_csv('data_processed/df_317.csv')
mayo_df =  pd.read_csv('data_processed/df_mayo.csv')
g_frts=[]
for k in df_317.columns:
    if k.startswith('g__'):
        g_frts.append(k)
#
#
seed_value = 2023
np.random.seed(seed_value)
models = {
        'LightGBM_all':LGBMClassifier(
                                  boosting_type='gbdt',
                                  objective='binary',
                                  metrics='auc',
                                  min_child_samples=3,
                                  num_leaves = 2 ** 5-1,
                                  max_depth = 5,
                                  learning_rate = 0.08
                                 ),
        
        'LightGBM_top6':LGBMClassifier(
                                  boosting_type='gbdt',
                                  objective='binary',
                                  metrics='auc',
                                  min_child_samples=3,
                                  num_leaves=2 ** 5-1,
                                  max_depth=5,
                                  learning_rate=0.05
                                 )
}
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=2024) 
#
model_pre_dict={}#把打分结果存起来
plt.figure(figsize=auc_figsize, dpi=200)
cache_auc_x = {}
for model_name, model in models.items():
    select_frts=g_frts[::]
    if fold != 3:
        continue
    #
    print('len(select_frts):', len(select_frts))
    df_cancer = df_317[:]
    train_x = df_cancer[select_frts].copy()
    train_y = df_cancer["MPN"]
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_x)
    print(train_x.shape,train_y.shape)
    # train_x = train_x.values
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold=0
    for train_index, test_index in kf.split(X_train_scaled, train_y):
        fold+=1
        
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = train_y[train_index], train_y[test_index]
        
        df_cancer[['#OTU ID']+select_frts+['MPN']].to_csv('train_df_cancer.csv',index=False)
        #拟合模型
        model.fit(X_train_fold, y_train_fold)
        y_scores = model.predict_proba(X_test_fold)[:, 1]
        model_pre_dict[model_name]=[y_scores.tolist(),y_test_fold.values.tolist()]
        fpr, tpr, thresholds = roc_curve(y_test_fold, y_scores)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        #print("model:{} roc_auc:{}".format(model_name,roc_auc))
        if model_name=='LightGBM_top6':
            plt.plot(fpr, tpr, lw=4.0, label='Top6 Microbial Species ' + ' (AUC = %0.3f)' % (roc_auc),
                     color = 'tab:cyan',
                    )
            
            
        if model_name=='LightGBM_all':
            plt.plot(fpr, tpr, lw=4.0, label='All Microbial Species' + ' (AUC = %0.3f)' % (roc_auc),
                     color = '#9b66fe')
            

        
#
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate',fontsize=20)
# plt.ylabel('True Positive Rate',fontsize=20)
plt.xlabel('1-Specificity',fontsize=20)
plt.ylabel('Sensitivity',fontsize=20)
plt.title('Receiver Operating Characteristic',fontsize=20,y=1.01)

#plt.title('Receiver Operating Characteristic',fontsize=20)
plt.legend(loc="lower right")
# 添加虚线格子
plt.grid(linestyle='dotted')
# 调整坐标轴线粗细
plt.rcParams['axes.linewidth'] = 2.5 # 默认是0.8
ax = plt.gca()
ax.spines['top'].set_linewidth(2.5) # 设置上边框线粗细
ax.spines['right'].set_linewidth(2.5) # 设置右边框线粗细
ax.spines['bottom'].set_linewidth(2.5) # 设置下边框线粗细
ax.spines['left'].set_linewidth(2.5) # 设置左边框线粗细
#plt.show()
os.makedirs(f'画图/{ds}/cancer/',exist_ok=True)
plt.savefig(f'画图/{ds}/cancer/lgb_top6_auc_curve_{ds}.pdf', bbox_inches='tight')
#plt.savefig(f'画图/{ds}/cancer_auc_curve_{ds}.png', bbox_inches='tight')
#plt.show()