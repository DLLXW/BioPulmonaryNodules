
import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import sklearn

from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'Arial'

import warnings
warnings.filterwarnings("ignore")



def func_1(df):
    df=df.transpose()#转置
    df.columns = df.iloc[0]
    df=df[1:].reset_index()
    df.columns=['#OTU ID']+df.columns[1:].tolist()
    return df
#
def func_2(s):
    #
    match = re.match(r'^([a-zA-Z]+)', s)
    if match:
        prefix = match.group(1)
    else:
        raise ValueError('Error!!')
    return prefix
#
def func_3(x):
    #
    if x.startswith('C'):
        return 0
    else:
        return 1
    
train_df = pd.read_csv('data_processed/train_df.csv')
tongue_df = pd.read_csv('data_processed/前3批 舌苔asv_taxon_Genus.csv')
throat_df = pd.read_csv('data_processed/前3批 咽拭子 asv_taxon_Genus.csv')
tongue_df=func_1(tongue_df)
throat_df=func_1(throat_df)

throat_df['label'] = throat_df['#OTU ID'].apply(lambda x:func_3(x))
tongue_df['label'] = tongue_df['#OTU ID'].apply(lambda x:func_3(x))
oral_df = train_df[:len(train_df)]
oral_df['label'] = oral_df['#OTU ID'].apply(lambda x:func_3(x))
#
tongue_df['ID'] = tongue_df['#OTU ID'].str.replace(r'^[a-zA-Z]+', '', regex=True)
throat_df['ID'] = throat_df['#OTU ID'].str.replace(r'^[a-zA-Z]+', '', regex=True)
oral_df['ID'] = oral_df['#OTU ID'].str.replace(r'^[a-zA-Z]+', '', regex=True)
#
tongue_df = tongue_df.drop_duplicates(subset=['ID'], keep='first')
throat_df = throat_df.drop_duplicates(subset=['ID'], keep='first')
oral_df = oral_df.drop_duplicates(subset=['ID'], keep='first')
#
inner_id = oral_df[['ID']].merge(tongue_df[['ID']])
inner_id = inner_id[['ID']].merge(throat_df[['ID']])
#
tongue_df = tongue_df.merge(inner_id[['ID']]).sort_values('ID')
throat_df = throat_df.merge(inner_id[['ID']]).sort_values('ID')
oral_df = oral_df.merge(inner_id[['ID']]).sort_values('ID')
#
#
candidate_df={
    '口腔':oral_df,
    '舌苔':tongue_df,
    '咽拭子':throat_df,
}
zh_en={
    '口腔':['Saliva','tab:green'],
    '舌苔':['Tongue Coating','tab:olive'],
    '咽拭子':['Throat swab','tab:red'],
}

auc_figsize=(12,10)
ds = '20240520'
plt.figure(figsize=auc_figsize, dpi=200)
for key in candidate_df.keys():
    tmp_df = candidate_df[key]
    select_frts = []
    for col in tmp_df.columns:
        if col.startswith('g__'):
            tmp_df[col] = tmp_df[col].astype(int)
            select_frts.append(col)
    #
    train_x = tmp_df[select_frts].copy()
    train_y = tmp_df["label"]
    important_lst_sp=[]
    #
    cv_scores = []
    cv_rounds = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
    #print(train_x.shape)
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        if i!=2:
            continue
        tr_x = train_x.iloc[train_index]
        tr_y = train_y.iloc[train_index]
        te_x = train_x.iloc[test_index]
        te_y = train_y.iloc[test_index]
        #
        train_matrix = lightgbm.Dataset(tr_x, label=tr_y)
        test_matrix = lightgbm.Dataset(te_x, label=te_y)
        
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metrics':'auc',
            'min_child_samples':1,
            'num_leaves': 2 ** 5-1,
            'max_depth': 5,
            'learning_rate': 0.05,
            'seed': 20,
            'nthread': 4,
            'num_class': 1,
            'verbose': -1,
        }
        num_round=500
        #
        model = lightgbm.train(params,
                               train_matrix, 
                               num_round,
                               valid_sets=test_matrix, 
                               verbose_eval=False,
                               #feval=tpr_eval_score,
                               early_stopping_rounds=200
                              )
        important_lst_sp.append(model.feature_importance("gain"))
        #print(importance_list)
        pre = model.predict(te_x, num_iteration=model.best_iteration)
        #pre=pre.argmax(axis=1)
        cv_scores.append(round(roc_auc_score(te_y, pre),3))
        fpr, tpr, thresholds = roc_curve(te_y, pre)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=4.0, label=zh_en[key][0] + ' (AUC = %0.3f)' % (roc_auc), color=zh_en[key][1])
        #
    print("{} AUC：{} Average AUC：{:.4}".format(key,[round(i,4) for i in cv_scores],np.mean(cv_scores)))
    #print("val_std:", np.std(cv_scores))

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
plt.savefig(f'画图/{ds}/cancer/Saliva_Tongue_Throat_curve_{ds}.pdf', bbox_inches='tight')
#plt.show()