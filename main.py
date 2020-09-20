from itertools import product

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def plot_result(x_transformed, y, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple']
    for c, label in zip(colors, digits.target_names):
        ax.scatter(x_transformed[y == int(label), 0], x_transformed[y == int(label), 1], color=c, label=label)
        ax.legend()
        ax.set_title(title, fontsize=16)
    plt.show()


class SNE:
    def __init__(self, n_components, perplexity, lr=0.01, n_epochs=100):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_epochs = n_epochs

    def _compute_perplexity_from_sigma(self, data_matrix, center_idx, sigma):
        similarities = self._similarity(data_matrix[center_idx, :], data_matrix, sigma, "h")
        p = similarities / similarities.sum()
        shannon = - (p[p != 0] * torch.log2(p[p != 0])).sum()  # log(0)回避
        perp = 2 ** shannon.item()
        return perp

    def _search_sigmas(self, data_matrix):
        sigmas = torch.zeros(self.N)
        sigma_range = np.arange(0.1, 0.6, 0.1)
        for i in tqdm(range(self.N), desc="search sigma"):
            perps = np.zeros(len(sigma_range))
            for j, sigma in enumerate(sigma_range):
                perp = self._compute_perplexity_from_sigma(data_matrix, i, sigma)
                perps[j] = perp
            best_idx = (np.abs(perps - self.perplexity)).argmin()
            best_sigma = sigma_range[best_idx]
            sigmas[i] = best_sigma
        return sigmas

    def _similarity(self, x1, x2, sigma, mode):
        # SNEでは高次元でも低次元でも正規分布を用いる
        return torch.exp(- ((x1 - x2) ** 2).sum(dim=1) / 2 * (sigma ** 2))

    def _compute_similarity(self, data_matrix, sigmas, mode):
        similarities = torch.zeros((self.N, self.N))
        for i in range(self.N):
            s_i = self._similarity(data_matrix[i, :], data_matrix, sigmas[i], mode)
            similarities[i] = s_i
        return similarities

    def _compute_cond_prob(self, similarities, mode):
        # SNEではmodeにより類似性の計算変わらない
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i in range(self.N):
            p_i = similarities[i] / similarities[i].sum()
            cond_prob_matrix[i] = p_i
        return cond_prob_matrix

    def fit_transform(self, X):
        self.N = X.shape[0]
        X = torch.tensor(X)

        # 1. yをランダムに初期化
        y = torch.randn(size=(self.N, self.n_components), requires_grad=True)
        optimizer = optim.Adam([y], lr=self.lr)

        # 2. 高次元空間の各データポイントに対応する正規分布の分散を指定されたperplexityから求める
        sigmas = self._search_sigmas(X)

        # 3. 高次元空間における各データポイント間の類似性を求める。
        X_similarities = self._compute_similarity(X, sigmas, "h")
        p = self._compute_cond_prob(X_similarities, "h")

        # 4. 収束するまで以下を繰り返し
        loss_history = []
        for i in tqdm(range(self.n_epochs), desc="fitting"):
            optimizer.zero_grad()
            y_similarities = self._compute_similarity(y, torch.ones(self.N) / (2 ** (1/2)), "l")
            q = self._compute_cond_prob(y_similarities, "l")

            kl_loss = (p[p != 0] * (p[p != 0] / q[p != 0]).log()).mean()  # log(0)回避
            kl_loss.backward()
            loss_history.append(kl_loss.item())
            optimizer.step()

        plt.plot(loss_history)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        return y.detach().numpy()


class TSNE(SNE):
    def _similarity(self, x1, x2, sigma, mode):
        if mode == "h":
            return torch.exp(- ((x1 - x2) ** 2).sum(dim=1) / 2 * (sigma ** 2))
        if mode == "l":
            return (1 + ((x1 - x2) ** 2).sum(dim=1)) ** (-1)

    def _compute_cond_prob(self, similarities, mode):
        cond_prob_matrix = torch.zeros((self.N, self.N))
        for i in range(self.N):
            p_i = similarities[i] / similarities[i].mean()
            cond_prob_matrix[i] = p_i

        if mode == "h":
            cond_prob_matrix = (cond_prob_matrix + torch.t(cond_prob_matrix)) / 2
        return cond_prob_matrix


if __name__ == "__main__":
    digits = load_digits()
    # only use top 100 samples for faster computation
    X, y = digits.data[:200, :], digits.target[:200]

    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    plot_result(X_pca, y, "PCA")

    sne = SNE(n_components=2, perplexity=50, n_epochs=200, lr=0.1)
    X_sne = sne.fit_transform(X_sc)
    plot_result(X_sne, y, "SNE")

    tsne = TSNE(n_components=2, perplexity=50, n_epochs=500, lr=0.1)
    X_tsne = tsne.fit_transform(X_sc)
    plot_result(X_tsne, y, "t-SNE")