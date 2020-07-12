import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import pandas as pd
sns.set_style("whitegrid")

def make_violins(df, x='method', y='rank', save=None, show=True):
    ax = sns.violinplot(x=x, y=y, data=df, color='0.8', bw=.1)
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)
    sns.stripplot(data=df, x=x, y=y, jitter=True, s=1, alpha=0.5)
    sns.despine()
    plt.ylim(-14, -3)
    plt.xlabel('Samples')
    plt.ylabel('Docking scores')
    if not save is None:
        plt.savefig(save, format="pdf")
    if show:
        plt.show()

    pass

df = pd.read_csv('violin_data.csv')
make_violins(df.loc[df['score'] < -3], x='subset', y='score', save = 'violin2.pdf')

zinc = df[df['subset']=='ZINC']
gianni = df[df['subset']=='Skalic et al.']
opt = df[df['subset']=='OptiMol']
mult = df[df['subset']=='OptiMol-multiobj']

pct_zinc = zinc[zinc['score']<-10].shape[0]/zinc.shape[0]


pct_opt = opt[opt['score']<-10].shape[0]/opt.shape[0]