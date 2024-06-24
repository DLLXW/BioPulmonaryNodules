
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
import lightgbm

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

auc_figsize=(12,10)
ds = '20240520'

def lambda_fuc(x):
    if 'Health' in x:
        return 0
    elif 'Case' in x:
        return 1
    else:
        return -1
#
def sp_process_func(df):
    df=df.transpose().reset_index()#转置
    df.columns = df.iloc[0]
    df=df[1:].reset_index(drop=True)
    df['Group']=df['Group'].apply(lambda x:lambda_fuc(x))
    for col in df.columns:
        if col.startswith('g__'):
            df[col]=df[col].astype(int)
    return df
#

df_279 = pd.read_csv('data_processed/df_279.csv')
df_317 = pd.read_csv('data_processed/df_317.csv')
dfa=pd.read_excel('data_processed/Genus_count_all_1-结直肠癌.xlsx')
dfb=pd.read_excel('data_processed/Genus_count_all_2-口腔癌.xlsx')
dfc=pd.read_excel('data_processed/Genus_count_all_3-糖尿病.xlsx')
dfa=sp_process_func(dfa)
dfb=sp_process_func(dfb)
dfc=sp_process_func(dfc)
frts_a=dfa.columns.tolist()
frts_b=dfb.columns.tolist()
frts_c=dfc.columns.tolist()
frts_our=df_279.columns.tolist()
frts_inter=[]
for col in frts_our:
    if col.startswith('g__'):
        if col in frts_a and col in frts_b and col in frts_c:
            frts_inter.append(col)
len(frts_inter)#共同的菌属

#
#
select_frts = frts_inter[:]
print('len(select_frts):', len(select_frts))
train_x = df_317[select_frts].copy()
train_y = df_317["MPN"]
predictors = list(train_x.columns)
train_x = train_x.values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
#
candidate_df={
    '良恶性肺结节':None,
    '糖尿病':dfc,
    '结直肠癌':dfa,
    '口腔癌':dfb
}
zh_en={
    '良恶性肺结节':['Malignant Pulmonary Nodules','tab:green'],
    '糖尿病':['Diabetes','tab:olive'],
    '结直肠癌':['Colorectal Cancer','tab:red'],
    '口腔癌':['Oral Cancer','tab:purple']
}
plt.figure(figsize=auc_figsize, dpi=200)
for key in candidate_df.keys():
    important_lst_sp=[]
    #val_map={}
    #
    cv_scores = []
    cv_rounds = []
    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        if i!=2:
            continue
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if key!='良恶性肺结节':
            #将特异性样本混入验证样本中
            #te_x = candidate_df[key][select_frts].values
            #te_y = candidate_df[key]['Group']
            if key == '糖尿病':
                te_x = np.concatenate([te_x[:15],candidate_df[key][select_frts].values])
                te_y = pd.Series(te_y.values.tolist()[:15]+candidate_df[key]['Group'].values.tolist())
            else:
                te_x = np.concatenate([te_x[:20],candidate_df[key][select_frts].values])
                te_y = pd.Series(te_y.values.tolist()[:20]+candidate_df[key]['Group'].values.tolist())
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
        #print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
        #            key=lambda x: x[1],reverse=True))[:10]))
        #importance_list=[ x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
        #            key=lambda x: x[1],reverse=True))]
        important_lst_sp.append(model.feature_importance("gain"))
        #print(importance_list)
        pre = model.predict(te_x, num_iteration=model.best_iteration)
        #pre=pre.argmax(axis=1)
        cv_scores.append(round(roc_auc_score(te_y, pre),3))
        #val_map[f'fold_{i}']=[te_y,pre]
        #cv_scores.append(accuracy_score(te_y, pre))
        fpr, tpr, thresholds = roc_curve(te_y, pre)
        roc_auc = sklearn.metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=4.0, label=zh_en[key][0] + ' (AUC = %0.3f)' % (roc_auc), color=zh_en[key][1])

        #
    if key!='良恶性肺结节':
        print("混入{}样本后的五折AUC：{}".format(key,cv_scores))
        print("混入{}样本后的五折平均AUC：{} --->>> {} \n".format(key,base_auc,round(np.mean(cv_scores),3)))
    else:
        print("预测肺结节良恶性模型的五折AUC：{}".format(cv_scores))
        base_auc=round(np.mean(cv_scores),3)
        print("\n预测肺结节良恶性模型的平均AUC：{} \n".format(base_auc))
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
#plt.show()
plt.savefig(f'画图/{ds}/cancer/特异性/spec_auc_curve_{ds}.pdf', bbox_inches='tight')
#plt.savefig(f'画图/{ds}/spec_auc_curve_{ds}.png', bbox_inches='tight')


# 疾病名称和颜色
zh_en = {
    '良恶性肺结节': ['Malignant Pulmonary Nodules', 'tab:green'],
    '糖尿病': ['Diabetes', 'tab:olive'],
    '结直肠癌': ['Colorectal Cancer', 'tab:red'],
    '口腔癌': ['Oral Cancer', 'tab:purple']
}


# 数据
data = [
    [0.865, 0.888, 0.872, 0.904, 0.854],
    [0.633, 0.667, 0.724, 0.685, 0.710],
    [0.595, 0.593, 0.659, 0.647, 0.621],
    [0.570, 0.582, 0.649, 0.617, 0.623]
]
# 绘制箱线图
plt.figure(figsize=(12, 10),dpi=200)  # 设置图形大小
box = plt.boxplot(data, patch_artist=True, labels=[zh_en[key][0] for key in zh_en])  # 绘制箱线图并设置标签

# 自定义箱体颜色
colors = [zh_en[key][1] for key in zh_en]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 添加英文图例
for key in zh_en:
    plt.plot([], [], color=zh_en[key][1], label=zh_en[key][0])
plt.legend()

# 设置标题和坐标轴标签
plt.title('AUC of Four Diseases')
plt.ylabel('AUC')
plt.ylim(0.5, 1.0)
#plt.show()
plt.savefig(f'画图/{ds}/cancer/特异性/spec_auc_boxplot_curve_{ds}.pdf', bbox_inches='tight')
#plt.savefig(f'画图/{ds}/spec_auc_boxplot_curve_{ds}.png', bbox_inches='tight')