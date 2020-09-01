import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from eval.eval_utils import plot_one

rc = {'figure.figsize': (10, 5),
      'axes.facecolor': 'white',
      'axes.grid': True,
      'lines.linewidth': 2.5,
      'grid.color': '.8',
      'font.size': 12}
plt.rcParams.update(rc)


def figure_replicates(dir_path=('plot/big_newlr', 'plot/big_newlr2', 'plot/big_newlr3'),
                      ylim=(-12, -6), plot_best=False,
                      return_best=False,
                      use_norm_score=False, obj='logp',
                      successive=False):
    fig, ax = plt.subplots(1, 2)

    all_scores = list()
    all_news = list()
    all_iters = list()
    for i, exp in enumerate(dir_path):
        iterations, mus, stds, batch_size, newslist, title, best_scores, best_smiles = plot_one(exp,
                                                                                                use_norm_score,
                                                                                                obj,
                                                                                                successive=successive)
        all_iters.append(iterations)
        all_news.append(newslist)
        all_scores.append(mus)

        sns.lineplot(iterations, mus, ax=ax[0], label=f'Replicate {i + 1}')
        ax[1].plot(iterations, newslist)

    # # Get min iterations and crop
    # min_iter = min([len(its) for its in all_iters])
    # print(min_iter)
    # iterations = all_iters[0][:min_iter]
    # all_news = [np.array(news[:min_iter]) for news in all_news]
    # all_news = np.stack(all_news)
    # all_scores = [np.array(score[:min_iter]) for score in all_scores]
    # all_scores = np.stack(all_scores)
    #
    # score_mus, score_std = np.mean(all_scores, axis=0), np.std(all_scores, axis=0)
    # news_mus, news_std = np.mean(all_news, axis=0), np.std(all_news, axis=0)
    #
    # ax[0].fill_between(iterations, score_mus + score_std, score_mus - score_std, alpha=.25)
    # sns.lineplot(iterations, score_mus, ax=ax[0])
    # ax[1].plot(iterations, news_mus)

    ax[0].set_ylim(ylim[0], ylim[1])
    ax[0].set_xlim(1, iterations[-1] + 0.2)
    ax[1].set_ylim(0, batch_size + 100)
    sns.despine()
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Docking Score (kcal/mol)')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Novel samples')
    # ax[1].legend()
    fig.tight_layout(pad=2.0)
    fig.align_labels()
    plt.savefig("cbas_replicated.pdf", format="pdf")
    plt.show()


figure_replicates()
