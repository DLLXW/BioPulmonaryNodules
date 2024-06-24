import pandas as pd
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import shap
shap.initjs()  # notebook环境下，加载用于可视化的JS代码
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({'font.size': 18})
plt.rcParams['font.family'] = 'Arial'

import warnings
warnings.filterwarnings("ignore")


top6_final = ['g__Actinomyces','g__Rothia','g__Streptococcus',
              'g__Prevotella','g__Porphyromonas','g__Veillonella']

df_317 = pd.read_csv('data_processed/df_317.csv')
select_frts_shap = top6_final #kimi_gpt
#select_frts_shap.remove('g__Fusobacterium')
train_x = df_317[select_frts_shap].copy()
train_y = df_317["cancer"]
X_train, X_test, y_train, y_test = train_test_split(train_x,
                                                    train_y,
                                                    test_size=0.2,
                                                    stratify=train_y,
                                                    random_state=10)
#
model = LGBMClassifier(max_depth=7,
                      boosting_type='gbdt',
                      objective='binary',
                      learning_rate=0.05,
                      n_estimators=100,
                      num_leaves=2 ** 5-1
                     )
model.fit(X_train,y_train)
y_scores = model.predict_proba(X_test)[:,1]
#
auc = roc_auc_score(y_test, y_scores)
print(f"auc:{auc}")

explainer = shap.TreeExplainer(model)#model 为训练好的机器学习模型
shap_values = explainer.shap_values(X_test)  # 传入特征矩阵X，计算SHAP值
#
auc_figsize=(12,10)
ds = '20240520'
pos_idx = []
for i in range(len(y_test.values)):
    if y_test.values[i]==1:
        pos_idx.append(i)
#
for sample_index in pos_idx:
    shap_plt = shap.plots.force(explainer.expected_value[1], shap_values[1][sample_index]
                    ,X_test.iloc[sample_index]
                    ,matplotlib=True
                    ,show=False
                    ,figsize=(16,6)
                    #,link_color='green'
                    ,text_rotation=270)
    #
    shap_plt.savefig(f'画图/{ds}/cancer/shap/single/cancer_shap_plot_{sample_index}_{ds}.pdf', bbox_inches='tight')
#
    # # 对多个样本进行综合解释并绘制图像
fig = plt.figure(figsize=(20,5),dpi=200)
shap.summary_plot(shap_values[1], X_test
                    ,max_display=10
                    ,plot_type="dot")

fig.savefig(f'画图/{ds}/cancer/shap/all/cancer_shap_summary_plot.pdf', bbox_inches='tight')