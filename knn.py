import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import adjusted_mutual_info_score
import os
import pandas as pd
import glob

from preprocessing import data_preprocessing

def df_PCA(df1, df2, df3):
    """
    Processing PCA
    """
    df1.columns = df1.columns.astype(str)
    del df1[df1.columns[0]]

    df2.columns = df2.columns.astype(str)
    del df2[df2.columns[0]]

    df3.columns = df3.columns.astype(str)
    del df3[df3.columns[0]]

    pca1 = PCA(n_components=2)
    principal_components1 = pca1.fit_transform(df1)

    pca2 = PCA(n_components=2)
    principal_components2 = pca2.fit_transform(df2)

    pca3 = PCA(n_components=2)
    principal_components3 = pca3.fit_transform(df3)

    df_pca1 = pd.DataFrame(data=principal_components1, columns=['PC1', 'PC2'], index=df1.index)

    df_pca2 = pd.DataFrame(data=principal_components2, columns=['PC1', 'PC2'], index=df2.index)

    df_pca3 = pd.DataFrame(data=principal_components3, columns=['PC1', 'PC2'], index=df3.index)

    return df_pca1, df_pca2, df_pca3


def cal_kmeans(df_list, num_clusters=11):
    """
    calculate kmeans
    """
    for j, df in enumerate([df_pca1, df_pca2, df_pca3]):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(df[['PC1', 'PC2']])
        df['cluster'] = kmeans.labels_


def plot_clusters(df_list, plot_show, num_clusters=11):
    # Create a list of unique colors for each cluster
    colors = sns.color_palette('tab20', num_clusters)
    fig, axs = plt.subplots(1, len(df_list), figsize=(10*len(df_list), 10))

    # Create a separate scatter plot for each dataframe
    for j, df in enumerate(df_list):
        ax = axs[j]
        for i in range(num_clusters):
            ax.scatter(df.loc[df['cluster']==i, 'PC1'], df.loc[df['cluster']==i, 'PC2'], color=colors[i], s=50, label=f'Cluster {i+1}')
            ax.set_title(f'Scatter Plot after PCA - Dataset {j+1}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(loc='lower right')

        # Draw convex hulls
        for i in range(num_clusters):
            cluster_points = df.loc[df['cluster']==i, ['PC1', 'PC2']]
            hull = ConvexHull(cluster_points)
            ax.fill(cluster_points.iloc[hull.vertices, 0], cluster_points.iloc[hull.vertices, 1], color=colors[i], alpha=0.1)
    
    if plot_show:
        plt.show()


def ami_score(y_true, df_pca):
    """
    calculate ami_score
    """
    label_cluster_map = pd.crosstab(y_true, df_pca["cluster"])
    ami = adjusted_mutual_info_score(y_true, df_pca["cluster"])

    return ami




if __name__ == "__main__":
    path = "training_data/"
    df1, df2, df3 = data_preprocessing(path)

    y_true1 = df1["gt"]
    y_true2 = df2["gt"]
    y_true3 = df3["gt"]

    df_pca1, df_pca2, df_pca3 = df_PCA(df1, df2, df3)
    cal_kmeans([df_pca1, df_pca2, df_pca3], num_clusters=11)

    plot_show = False
    plot_clusters([df_pca1, df_pca2, df_pca3], plot_show, num_clusters=11)

    ami_1 = ami_score(y_true1, df_pca1)
    ami_2 = ami_score(y_true2, df_pca2)
    ami_3 = ami_score(y_true3, df_pca3)

    print("Adjusted Mutual Information:", ami_1, ami_2, ami_3)
