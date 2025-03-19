import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr

ftrDir = '1_feature_tumor_300'  # ftr in tumor region
RNA = pd.read_csv('RNA_4genes.csv') 

RNA = RNA.T
RNA.columns = RNA.iloc[0]  
RNA = RNA[1:]  

# WSI Ftr (mean)
averages = []
for filename in os.listdir(ftrDir):
    if filename.endswith('.npy'):
        file_path = os.path.join(ftrDir, filename)
        data = np.load(file_path) # 960列

        row_means = np.nanmean(data, axis=0)  # mean
        averages.append(row_means)

averages_df = pd.DataFrame(averages)
averages_df.index = [filename.split('_')[1][0:12] for filename in os.listdir(ftrDir) if filename.endswith('.npy')]

averages_df = averages_df.groupby(averages_df.index).mean()
averages_df.to_csv('WSIftr_tumor_300.csv')  # 364，960

# scale
averages_df = (averages_df - averages_df.min()) / (averages_df.max() - averages_df.min())
merged_df = pd.merge(RNA, averages_df, left_index=True, right_index=True, how='inner')
merged_df.shape

# association
gene_expressions = merged_df.iloc[:, :4]  # 0:960 are features，the last 4 are gene expression
features = merged_df.iloc[:, -960:]

rho_results = pd.DataFrame(index=features.columns, columns=gene_expressions.columns)
p_values = pd.DataFrame(index=features.columns, columns=gene_expressions.columns)

# 循环计算相关性
for feature in features.columns:
    for gene in gene_expressions.columns:
        
        valid_indices = features[feature].notna() & gene_expressions[gene].notna()
        if valid_indices.sum() > 1:  # 确保有足够的有效数据
            rho, p = pearsonr(features[feature][valid_indices], gene_expressions[gene][valid_indices])
            rho_results.loc[feature, gene] = rho
            p_values.loc[feature, gene] = p
        
        else:
            # 如果有效数据不足，设置为 NaN
            rho_results.loc[feature, gene] = np.nan
            p_values.loc[feature, gene] = np.nan
            
# 输出结果到 CSV
rho_results.to_csv('2_association/rho_results.csv')
p_values.to_csv('2_association/p_values.csv')