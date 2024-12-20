import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np
from matplotlib.ticker import StrMethodFormatter


X = [16, 32, 64, 128]
XX = np.log2(X)
Y = np.array([591.804, 1148.765, 2739.726, 4839.685])/33.313

YY = np.log2(Y)

print(YY)

df = pd.DataFrame(data={'P': XX, 'MPts/sec': YY})

p = sns.regplot(x='P', y='MPts/sec', data=df, scatter_kws={"color": "red", "s":75}, line_kws={"color": "black"}, ci=None)
p.set_xlabel("Number of Processors")
p.set_ylabel("Absolute Speedup")
p.set_xticks(XX)
p.set_xticklabels(X)
p.set_yticks(YY)
# p.set_yticklabels(Y)
p.set_yticklabels(StrMethodFormatter('{x:.2f}').format_ticks(Y)) # 2 decimal places
plt.savefig('report/speedup_2000.png')