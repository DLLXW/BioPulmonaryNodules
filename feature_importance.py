import numpy as np
import pandas as pd
import lightgbm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,log_loss,precision_recall_curve,average_precision_score
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
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
#plt.style.use('seaborn-paper')
import copy
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#
df_317 = pd.read_csv('data_processed/df_317.csv')
g_frts=[]
for k in df_317.columns:
    if k.startswith('g__'):
        g_frts.append(k)
#
important_lst_cancer=[]
val_map={}
#select_frts = [f for f in train_df.columns if f not in to_del]
select_frts = g_frts[::]
print('len(select_frts):', len(select_frts))
#
train_x = df_317[select_frts].copy()
train_y = df_317["cancer"]
print(train_x.shape)
predictors = list(train_x.columns)
train_x = train_x.values
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train = np.zeros((train_x.shape[0]))
cv_scores = []
cv_rounds = []

for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
    tr_x = train_x[train_index]
    tr_y = train_y[train_index]
    te_x = train_x[test_index]
    te_y = train_y[test_index]
    #te_x = train_df_nodules[select_frts]
    #te_y = train_df_nodules['MPN']
    train_matrix = lightgbm.Dataset(tr_x, label=tr_y)
    test_matrix = lightgbm.Dataset(te_x, label=te_y)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metrics':'auc',
        'min_child_samples':10,
        'num_leaves': 2 ** 6-1,
        'max_depth': 7,
        'learning_rate': 0.05,
        'seed': 2023,
        'nthread': 4,
        'num_class': 1,
        'verbose': -1,
    }
    num_round=2000
    #
    model = lightgbm.train(params,
                           train_matrix, 
                           num_round,
                           valid_sets=test_matrix, 
                           verbose_eval=100,
                           #feval=tpr_eval_score,
                           early_stopping_rounds=200
                          )
    #print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
    #            key=lambda x: x[1],reverse=True))[:10]))
    #importance_list=[ x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
    #            key=lambda x: x[1],reverse=True))]
    important_lst_cancer.append(model.feature_importance("gain"))
    #print(importance_list)
    pre = model.predict(te_x, num_iteration=model.best_iteration)
    #pre=pre.argmax(axis=1)
    cv_scores.append(roc_auc_score(te_y, pre))
    val_map[f'fold_{i}']=[te_y,pre]
    #cv_scores.append(accuracy_score(te_y, pre))
    #
    print("cv_score is:", cv_scores)
    #break
#
print("val_mean:", np.mean(cv_scores))
print("val_std:", np.std(cv_scores))
#
important_lst_cancer=np.array(important_lst_cancer)
important_lst_cancer=np.mean(important_lst_cancer,axis=0)
sns.set_context("paper")  # sns也可以设置使用paper风格
sns.set_context("talk", font_scale=1.0, rc={'line.linewidth':2.5})#字体/线宽
#
feature_imp_cancer = pd.DataFrame(sorted(zip(important_lst_cancer,df_317[select_frts].columns)), columns=['Value','Feature'])
feature_imp_cancer = feature_imp_cancer[::-1].reset_index(drop=True)
feature_top=feature_imp_cancer[:10]
feature_top['Gain Of The Split']=feature_top['Value']#.apply(lambda x: np.log(x+1))
feature_top['Log(Gain) Of The Split']=feature_top['Value'].apply(lambda x: np.log(x+1))
print(feature_top)
#综合多种树模型特征重要度
# train_x = df_317[select_frts].copy()
# train_y = df_317["MPN"]
# 将数据集划分为训练集和验证集（这里以验证集占比 20% 为例）
train_x, val_x, train_y, val_y = train_test_split(df_317[select_frts], df_317["cancer"],
                                                    test_size=0.2, random_state=seed_r)

topK=15
# 随机森林
rf = RandomForestClassifier(n_estimators=100,random_state=62)
rf.fit(train_x, train_y)
rf_importances = rf.feature_importances_
rf_top_indices = np.argsort(rf_importances)[-topK:]
rf_top_features = train_x.columns[rf_top_indices]
rf_val_y_pred_proba = rf.predict_proba(val_x)[:, 1]
rf_auc = roc_auc_score(val_y, rf_val_y_pred_proba)
#print("随机森林模型在验证集上的AUC值：", rf_auc)

# XgBoost
xgb = XGBClassifier(max_depth=7,
                    learning_rate=0.1,
                    n_estimators=100,eval_metric= 'error')
xgb.fit(train_x, train_y)
xgb_importances = xgb.feature_importances_
xgb_top_indices = np.argsort(xgb_importances)[-topK:]
xgb_top_features = train_x.columns[xgb_top_indices]
xgb_y_pred_proba = xgb.predict_proba(val_x)[:, 1]
xgb_auc = roc_auc_score(val_y, xgb_y_pred_proba)
#print("XGBoost模型的AUC值：", xgb_auc)

# LightGBM
lgb = LGBMClassifier(n_estimators=100)
lgb.fit(train_x, train_y)
lgb_importances = lgb.feature_importances_
lgb_top_indices = np.argsort(lgb_importances)[-topK:]
lgb_top_features = train_x.columns[lgb_top_indices]
lgb_val_y_pred_proba = lgb.predict_proba(val_x)[:, 1]
lgb_auc = roc_auc_score(val_y, lgb_val_y_pred_proba)
#print("LightGBM模型在验证集上的AUC值：", lgb_auc)

# 特征重要度归一化
rf_importances /= np.sum(rf_importances)
xgb_importances /= np.sum(xgb_importances)
lgb_importances = lgb_importances / np.sum(lgb_importances)

# 为每个模型的特征重要度赋予权重（这里假设权重分别为 0.4、0.3 和 0.3）
weighted_importances = (0.33 * rf_importances + 0.33 * xgb_importances + 0.33 * lgb_importances)

# 获取 top K 特征
topk_indices = np.argsort(weighted_importances)[::-1][:topK]
topk_features = np.array(select_frts)[topk_indices]
topk_importances = weighted_importances[topk_indices]
#
#-------------plot--------------
ds = 20240520
# 设置图像尺寸和分辨率
plt.figure(figsize=(30, 15), dpi=200)

feature_top=pd.DataFrame(columns=['topk_features','Gain Of The Split'])
feature_top['Feature']=topk_features
feature_top['Gain Of The Split']=topk_importances

# 调整颜色
# color_palette = sns.color_palette("coolwarm", len(feature_top))
# color_palette = color_palette[::-1]
#colors = [(0, '#E8F5E9'), (0.5, '#A5D6A7'),(1, '#81C784')]  # 绿色
#colors = [(0, '#C9F2FD'), (0.5, '#79BEE6'), (1, '#2E86C0')]  # 蓝色
colors = [(0, '#E6F7FF'), (0.33, '#C9F2FD'), (0.66, '#79BEE6'), (1, '#2E86C0')]  # 蓝色
colors = [(0, '#C9F2FD'), (0.66, '#C9F2FD'), (1, '#79BEE6')]  # 蓝色
#colors = [(0, '#E6F7FF'), (0.25, '#C9F2FD'), (0.5, '#79BEE6'), (0.75, '#FFCDD2'), (1,'#FA8072')] #
#
#['g__Actinomyces','g__Rothia','g__Streptococcus','g__Prevotella','g__Porphyromonas','g__Veillonella']
#
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors,N=len(feature_top))
colors_list = [cmap(i) for i in range(cmap.N)][::-1]
# 创建自定义调色板
custom_palette = sns.color_palette(colors_list)
sns.barplot(x="Gain Of The Split", y="Feature", data=feature_top.sort_values(by="Gain Of The Split", ascending=False),
            palette=colors_list)

plt.title('Feature Importance (Top 6 Features)', fontsize=30, y=1.01)  # 设置标题字体大小为16
#plt.grid(True, axis='x',linestyle='-', linewidth=0.5, color='lightgrey')  # 显示水平网格线
#plt.grid(True, axis='y',linestyle='-', linewidth=0.5, color='lightgrey')  #
plt.xlabel('Feature Importance', fontsize=30) # x轴标签字体大小
plt.ylabel('Microbial Species', fontsize=30) # y轴标签字体大小
plt.xticks(fontsize=30) # x轴刻度标签字体大小
plt.yticks(fontsize=30) # y轴刻度标签字体大小
plt.tight_layout()
# 保存图像
plt.savefig(f'画图/{ds}/cancer/cancer_feature_importance_viridis_{ds}.pdf', bbox_inches='tight')

#小提琴图
mpn_global_top = topk_features[::]
plt.rcParams.update({'font.size': 24})

df_selected = df_317[mpn_global_top+['MPN','cancer']]
#df_selected = df_selected[~ ((df_selected['cancer']==1) & (df_selected['MPN']==0))].reset_index(drop=True)
#
df_selected['MBPN'] = df_selected['cancer'].apply(lambda x:'MPN' if x==1 else 'BPN')

#sns.set_style(style="whitegrid") 
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16,10), dpi=300)


sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2.2)

# 恢复到默认的整体风格和字体大小
# sns.set_theme()  # 恢复整体风格为默认值
# sns.set_context("notebook")  # 恢复字体大小为默认值

# 逐个绘制箱线图
for i, microbe in enumerate(mpn_global_top):
    row = i // 2
    col = i % 2
    #sns.boxplot(x='PN', y=microbe, data=df_selected, ax=ax[row, col],color=None, linewidth=2.5)
    sns.violinplot(x='MBPN', y=microbe, data=df_selected,
                   palette = {'MPN': '#d95848', 'BPN': '#00a1d5'},
                   ax=ax[row, col])
    ax[row, col].set_xlabel('')  # 移除 x 轴标签
    ax[row, col].set_ylabel(microbe)
    ax[row, col].set_title('')  # 设置子图标题为空字符串
# 调整子图之间的距离和布局
plt.tight_layout()
#plt.tight_layout(pad=2.0)
# 调整坐标轴线粗细
# plt.rcParams['axes.linewidth'] = 2.5 # 默认是0.8
# ax = plt.gca()
# ax.spines['top'].set_linewidth(2.5) # 设置上边框线粗细
# ax.spines['right'].set_linewidth(2.5) # 设置右边框线粗细
# ax.spines['bottom'].set_linewidth(2.5) # 设置下边框线粗细
# ax.spines['left'].set_linewidth(2.5) # 设置左边框线粗细
#plt.show()
plt.savefig(f'画图/{ds}/cancer/cancer_violinplot_{ds}.pdf', bbox_inches='tight')
#plt.savefig(f'画图/{ds}/cancer_violinplot_{ds}.png', bbox_inches='tight')
plt.rcParams.update({'font.size': 18})

#------------------KDE--------------------
# 设置颜色
colors = {'BPN': 'lightblue', 'MPN': 'lightcoral'}

# 创建子图布局
fig, axes = plt.subplots(3, 2, figsize=(20, 8),dpi=300)

# 遍历每个细菌特征，画出kde分布图
for i, microbe in enumerate(mpn_global_top):
    row = i // 2
    col = i % 2
    
    # 画 BPN 的 KDE 图
    sns.kdeplot(data=df_selected[df_selected['MBPN']=='BPN'][microbe], color='#00a1d5', ax=axes[row, col]
                , label='BPN', linewidth=4)
    
    # 画 MPN 的 KDE 图
    sns.kdeplot(data=df_selected[df_selected['MBPN']=='MPN'][microbe], color='#d95848', ax=axes[row, col]
                , label='MPN', linewidth=4)
    
    axes[row, col].tick_params(labelsize=20)  # 设置坐标轴标签大小
    axes[row, col].set_xlabel('')  # 移除 x 轴标签
    axes[row, col].set_ylabel(microbe, fontsize=20)  # 设置 y 轴标签大小
    axes[row, col].set_title('')  # 设置子图标题为空字符串
    axes[row, col].legend()  # 添加图例

plt.tight_layout()
plt.savefig(f'画图/{ds}/cancer/cancer_kde_{ds}.pdf', bbox_inches='tight')

#------------------Heatmap--------------------
# 计算相关性矩阵
df_selected_copy = df_selected.drop('cancer',axis=1)
correlation_matrix = df_selected_copy.corr(method='spearman')#pearson spearman
# 绘制相关性矩阵的热力图
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',annot_kws={"size": 20})
plt.title('Correlation Matrix', fontsize=20, y=1.01)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(f'画图/{ds}/cancer/cancer_heatmap_{ds}.pdf', bbox_inches='tight')
#plt.savefig(f'画图/{ds}/cancer_heatmap_{ds}.png', bbox_inches='tight')
#plt.show()
