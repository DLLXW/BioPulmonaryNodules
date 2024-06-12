import numpy as np
import pandas as pd
import os

def parse_label(x):
    if x.startswith('ZK'):
        return 1
    else:
        return 0
#
def parse_label_v1(x):
    if x.startswith('ZK') or x in false_neg:
        return 1
    else:
        return 0

if __name__=="__main__":

    false_neg = ['K0923x3',
            'K0610x5',
            'K0618x1',
            'K0610xB',
            'K0710x1',
            'K1009x3',
            'K0625x2',
            'K30324x3',
            'K1029x1',
            'K1112x1',
            'K30415x1',
            'K0612x2',
            'K30326x1',
            'K0813x2',
            'K30303x3']
    
    print(len(false_neg))
    df_279 = pd.read_csv('data/df_279.csv')
    df_317 = pd.read_csv('data/df_317.csv')
    df_base = pd.read_csv('data/cancer_213_20231213.csv')
    df_base.rename(columns={'cancer': 'MPN'}, inplace=True)
    df_base=df_base.rename(columns={'性别': "gender",
                                '年龄': "age",
                                '个人肿瘤史':'personal_cancer_history',
                                '家族肿瘤史':'family_cancer_history',
                                '吸烟史':'smoke'})
    #
    df_saliva=pd.read_excel('data/482_saliva.xlsx')
    df_saliva=df_saliva.transpose()#转置
    df_saliva.columns = df_saliva.iloc[0]
    df_saliva=df_saliva[1:].reset_index(drop=True)
    df_saliva.columns=['#OTU ID']+df_saliva.columns[1:].tolist()
    #
    df_279['MPN']=df_279['#OTU ID'].apply(lambda x:parse_label(x))
    df_279['cancer']=df_279['#OTU ID'].apply(lambda x:parse_label_v1(x))
    for col in df_279.columns:
        if col.startswith('g__'):
            df_279[col]=df_279[col].astype(int)
    #
    df_317['MPN']=df_317['#OTU ID'].apply(lambda x:parse_label(x))
    df_317['cancer']=df_317['#OTU ID'].apply(lambda x:parse_label_v1(x))
    for col in df_317.columns:
        if col.startswith('g__'):
            df_317[col]=df_317[col].astype(int)
    #
    need_drop=[]
    for k in df_saliva.columns:
        if k in df_base.columns:
            if k!='#OTU ID':
                need_drop.append(k)
    df_base=df_base.drop(need_drop,axis=1)
    #
    g_frts=[]
    for k in df_279.columns:
        if k in df_saliva.columns and k.startswith('g__'):
            g_frts.append(k)
    len(g_frts)
    #
    df=pd.merge(df_base,df_saliva,on='#OTU ID')
    # 找到数值列
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    # 找到object列
    object_cols = df.select_dtypes(include='object').columns
    #
    #除了 '#OTU ID','影像资料','危险分层' 这仨，其余的object列其实本身都是数字，所以可以直接转换成int类型
    for col in object_cols:
        if col not in ['#OTU ID','影像资料','危险分层']:
            df[col]=df[col].astype(int)
    object_cols = ['#OTU ID','影像资料','危险分层']
    #
    df[numeric_cols] = df[numeric_cols].fillna(-999)#数值为空的列，用-999来填充
    df['影像资料'] = df['影像资料'].fillna('')#文本用空字符串填充
    df_emmpy = pd.concat((df[df['危险分层'].isnull()],df[df['危险分层']=='不清']))
    df_clean = df[(df['危险分层'].notnull()) & (df['危险分层']!='不清')].reset_index(drop=True)
    df_clean['危险分层']=df_clean['危险分层'].astype(int)
    #
    train_df = df_clean
    train_df['PN']=(train_df['危险分层']>0).astype(int)
    train_df_nodules=train_df[train_df['PN']>0].reset_index(drop=True)
    train_df_nodules['MPN']=train_df_nodules['MPN'].astype(int)
    #
    os.makedirs('data_processed/', exist_ok=True)
    pd.to_csv('data_processed/train_df.csv', index=False)
    pd.to_csv('data_processed/train_df_nodules.csv', index=False)
    pd.to_csv('data_processed/df_279.csv', index=False)
    pd.to_csv('data_processed/df_317.csv', index=False)