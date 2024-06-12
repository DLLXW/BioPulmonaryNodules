
import os
ds='20240520'
save_dir = f'画图/{ds}'
os.makedirs(save_dir,exist_ok=True)
os.makedirs(save_dir+f'/cancer/',exist_ok=True)
os.makedirs(save_dir+f'/cancer/shap/',exist_ok=True)
os.makedirs(save_dir+f'/cancer/特异性/',exist_ok=True)
os.makedirs(save_dir+f'/nodules/',exist_ok=True)
os.makedirs(save_dir+f'/nodules/shap',exist_ok=True)