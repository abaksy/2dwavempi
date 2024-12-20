import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

X = [2, 4, 8, 16, 16, 32, 64, 128, 128, 192, 256, 384]
Y = np.array([76.590, 157.146, 305.018, 616.903, 591.804, 1148.765, 2739.726, 4839.685, 4875.076,7774.538, 9353.079, 13377.926])
Y_nocomm = np.array([89.154, 180.310, 340.657, 663.79, 648.246, 1242.429, 2875.850, 4932.182, 4969.985, 7929.958, 9764.036, 14427.412])

df = pd.DataFrame(data={'P': X, 'std': Y, 'noComm': Y_nocomm})
df_melt = df.melt(id_vars="P", value_vars=["std", "noComm"], var_name="type", value_name='Performance (MPts/s)')
p = sns.pointplot(data=df_melt, x='P', y='Performance (MPts/s)', hue='type')

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))

# for i in range(len(ax)):
#     df = pd.DataFrame(data={'P': X[4*i:4*(i+1)], 'std': Y[4*i:4*(i+1)], 'noComm': Y_nocomm[4*i:4*(i+1)]})
#     df_melt = df.melt(id_vars="P", value_vars=["std", "noComm"], var_name="type", value_name='Performance (MPts/s)')
#     # print(df_melt)
#     p = sns.pointplot(data=df_melt, x='P', y='Performance (MPts/s)', hue='type', ax=ax[i])
plt.show()