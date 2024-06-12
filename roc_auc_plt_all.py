import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,log_loss,precision_recall_curve,average_precision_score
from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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

#恶性概率
def mayo(row):
    age = row['age']
    smoke = row['smoke']
    history = row['history']
    scale = row['scale']
    density = row['density']
    loc = row['loc']
    if scale>1000:
        scale=2.5
    x=-6.8272+0.0391*age + 0.7917*smoke +1.3388*history +0.1274*scale + 1.0407*density + 0.7838*loc
    
    p = np.exp(x)/(1 + np.exp(x))
    return p

custom_colors = {
    'Mayo': 'tab:blue',
    'Mayo-BioEnhanced': 'tab:orange',
    'Decision Tree': 'tab:blue',
    'K-Nearest Neighbours': 'tab:orange',
    'Support Vector Machine': 'tab:green',
    'Logistic Regression': 'tab:red',
    'Multilayer Perceptron': 'tab:purple',
    'Naive Bayes': 'tab:brown',
    'Random Forest': 'tab:pink',
    'GBDT': 'tab:olive',
    'LightGBM': 'tab:cyan',
    'XGBoost': '#9467bd',  # 使用Hex颜色代码
    'AdaBoost': '#7f7f7f',  # 使用Hex颜色代码
    'CatBoost': '#bcbd22',  # 使用Hex颜色代码
    'Extra Trees': '#d62728',  # 使用Hex颜色代码
    'Gradient Boosting': '#9467bd',  # 使用Hex颜色代码
    'Lasso Regression': '#8c564b',  # 使用Hex颜色代码
    'Ridge Regression': '#2ca02c'  # 使用Hex颜色代码
}
#

auc_figsize=(12,10)
ds = '20240520'
df_317 = pd.read_csv('data_processed/df_317.csv')
mayo_df =  pd.read_csv('data_processed/df_mayo.csv')
g_frts=[]
for k in df_317.columns:
    if k.startswith('g__'):
        g_frts.append(k)
#
seed_value = 2023
np.random.seed(seed_value)
models = {
        #'Mayo':mayo,
        #'Decision Tree': DecisionTreeClassifier(),
        #'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=10,leaf_size=30),
        'Support Vector Machine': SVC(probability=True,C=1.0),
        'Logistic Regression': LogisticRegression(),
        'Multilayer Perceptron':MLP(),
        'Naive Bayes': BernoulliNB(alpha=1.0),#GaussianNB,MultinomialNB,BernoulliNB(alpha=1.2)
        'Random Forest': RandomForestClassifier(n_estimators=80,min_samples_split=5,random_state=2023),#,min_samples_split=5
        'GBDT':GradientBoostingClassifier(max_depth=5,
                                learning_rate=0.1,
                                n_estimators=300),
        'LightGBM':LGBMClassifier(
                                  boosting_type='gbdt',
                                  objective='binary',
                                  metrics='auc',
                                  min_child_samples=3,
                                  num_leaves=2 ** 5-1,
                                  max_depth=5,
                                  learning_rate=0.05
                                 )
}
#
select_frts=g_frts[::]
print('len(select_frts):', len(select_frts))
df_cancer = df_317[:]
train_x = df_cancer[select_frts].copy()
train_y = df_cancer["MPN"]
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_x)
print(train_x.shape,train_y.shape)
# train_x = train_x.values
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=2024) 
#
model_pre_dict={}#把打分结果存起来
num_models = len(models)
fig, axes = plt.subplots(num_models, 2, figsize=(25, 10*num_models), dpi=200)  # 根据模型数量调整画布大小
cnt = 0
for model_name, model in models.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    fold=0
    for train_index, test_index in kf.split(X_train_scaled, train_y):
        fold+=1
        if fold!=3:
            continue
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = train_y[train_index], train_y[test_index]
        np.save("train_index_cancer.npy",train_index)
        np.save("test_index_cancer.npy",test_index)
        df_cancer[['#OTU ID']+select_frts+['MPN']].to_csv('train_df_cancer.csv',index=False)
        #拟合模型
        if model_name=='Mayo':
            mayo_df_test=mayo_df.merge(df_cancer.iloc[test_index][['#OTU ID']],on='#OTU ID')
            if mayo_df_test.shape[0]<len(test_index):
                diff=len(test_index)-mayo_df_test.shape[0]
                diff_tmp = mayo_df[~mayo_df['#OTU ID'].isin(mayo_df_test['#OTU ID'])]
                mayo_df_test = pd.concat([mayo_df_test,diff_tmp.sample(n=diff)]).reset_index(drop=True)
            y_scores = []
            for _,row in mayo_df_test.iterrows():
                y_scores.append(mayo(row))
            #
            y_scores=np.array(y_scores)
            y_test_fold=mayo_df_test['MPN']
        else:
            model.fit(X_train_fold, y_train_fold)
            y_scores = model.predict_proba(X_test_fold)[:, 1]
        model_pre_dict[model_name]=[y_scores.tolist(),y_test_fold.values.tolist()]
        #
        fpr, tpr, thresholds = roc_curve(y_test_fold, y_scores)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        #
        #混淆矩阵画图
        precision, recall, thresholds = precision_recall_curve(y_test_fold, y_scores)
        ap = average_precision_score(y_test_fold, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores, copy=True)
        max_f1_score_idx = np.argmax(f1_scores)
        max_f1_score = f1_scores[max_f1_score_idx]
        max_precision = precision[max_f1_score_idx]
        max_recall = recall[max_f1_score_idx]
        threshold = thresholds[max_f1_score_idx]
        #
        threshold_optimal = thresholds[max_f1_score_idx - 1]  # 调整此处
        # 使用最佳阈值来预测测试集的标签
        y_pred_optimal = (y_scores >= threshold_optimal).astype(int)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test_fold, y_pred_optimal)
        ## 归一化混淆矩阵
        cm_normalized = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3) 

        # 创建画布和子图
        ax_cm = axes[cnt, 0]

        # 子图1: 绘制混淆矩阵
        colors = [(0, 'white'), (0.25, '#F3E5F5'), (0.5, '#CE93D8'), (1, '#6A1B9A')]  # 紫色
        colors = [(0, 'white'), (0.25, '#C9F2FD'), (0.5, '#79BEE6'), (1, '#2E86C0')]  # 蓝色
        colors = [(0, 'white'), (0.25, '#C9F2FD'), (0.5, '#79BEE6'), (0.75, '#2E86C0'),(1, '#003271')]  # 蓝色
        #colors = [(0, 'white'), (0.25, '#E8F5E9'), (0.5, '#A5D6A7'),(1, '#81C784')] #绿色
        #
        #colors = cmap_dict[custom_colors[model_name]]
        
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        
        #cax = ax_cm.matshow(cm_normalized, cmap=plt.cm.Blues,)
        cax = ax_cm.matshow(cm_normalized, cmap=cmap)
        fig.colorbar(cax, ax=ax_cm)
        ax_cm.set_xlabel('Predicted labels', fontsize=25)
        ax_cm.set_ylabel('True labels', fontsize=25)
        ax_cm.set_xticklabels(['', 'BPN', 'MPN'], fontsize=25)  # 设置字体大小
        ax_cm.set_yticklabels(['', 'BPN', 'MPN'], fontsize=25)  # 设置字体大小
        ax_cm.tick_params(axis='both', which='major', labelsize=25)  # 控制刻度标签的字体大小
        #ax_cm.set_title('Confusion Matrix', fontsize=15)  # 设置标题字体大小

        # 在格子上添加数字
        for (i, j), val in np.ndenumerate(cm_normalized):
            ax_cm.text(j, i, f'{val}'
                       , ha='center'
                       , va='center'
                       , fontsize=25
                       , color=custom_colors[model_name])

        # 子图2: 绘制ROC曲线
        fpr, tpr, _ = roc_curve(y_test_fold, y_scores)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        ax_roc = axes[cnt, 1]
        ax_roc.plot(fpr, tpr, color=custom_colors[model_name], lw=5, label=f'{model_name} (area = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate', fontsize=25)
        ax_roc.set_ylabel('True Positive Rate', fontsize=25)
        ax_roc.tick_params(axis='both', which='major', labelsize=28)  # 控制刻度标签的字体大小
        #ax_roc.set_title('Receiver Operating Characteristic', fontsize=15)  # 设置标题字体大小
        ax_roc.legend(loc="lower right", fontsize=25)  # 设置图例字体大小
        
        #fig.suptitle(model_name, fontsize=15)  # 添加总标题
        cnt+=1
#
plt.tight_layout()
plt.savefig(f'画图/{ds}/cancer/all_models_combined_green.pdf', bbox_inches='tight')