

import numpy as np
import pandas as pd


def process_mayo(df):
    history = df['个人肿瘤史'].values
    smoke = df['烟酒史'].values
    scale = df['长径'].values
    loc = df['位置'].values
    density = df['密度'].values
    #
    history_lst = []
    for k in history:
        try:
            if '癌' in k or '瘤' in k or 'ca' in k or 'CA' in k:
                history_lst.append(1)
            else:
                history_lst.append(0)
        except:
            history_lst.append(0)
    #
    smoke_lst = []
    for k in smoke:
        try:
            if '烟' in k:
                smoke_lst.append(1)
            else:
                smoke_lst.append(0)
        except:
            smoke_lst.append(0)
    #
    scale_lst = []
    for k in scale:
        try:
            scale_lst.append(float(k))
        except:
            scale_lst.append(0.0)
    #
    loc_lst = []
    for k in loc:
        try:
            if '上叶' in k:
                loc_lst.append(1)
            else:
                loc_lst.append(0)
        except:
            loc_lst.append(0)
    #
    density_lst = []
    for k in density:
        try:
            if '毛刺' in k:
                density_lst.append(1)
            else:
                density_lst.append(0)
        except:
            density_lst.append(0)
    #
    df['age']=df['年龄'].astype(float)
    df['history']=history_lst
    df['smoke']=smoke_lst
    df['scale']=scale_lst
    df['loc']=loc_lst
    df['density']=density_lst
    df=df.fillna(0)
    return df
#
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

if __name__=="__main__":
    # e为自然对数；
    # 年龄按数字计算；
    # 如果既往有吸烟史（无论是否已戒除）则为1，否则为0；
    # 如果5年内（含5年）有胸外肿瘤史则为1，否则为0；
    # 结节直径以毫米为单位计算；
    # 如果结节边缘有毛刺则为1，否则为0；
    # 如果肺结节定位在上叶则为1，否则为0。
    df_279 =  pd.read_csv('data_processed/df_279.csv')
    df_mayo = process_mayo(df_279)
    df_mayo.to_csv('data_processed/df_mayo.csv')