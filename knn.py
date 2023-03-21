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


def data_preprocessing(path):
    # combine data_po files into df1
    files_po = glob.glob(path + "*data_po*.csv")
    print(files_po)
    df1 = pd.DataFrame()
    for file in files_po:
        with open(file, 'r') as f:
            csv_string = f.read()

            data = csv_string
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
            df.drop(df.tail(1).index,inplace=True)

            temp_df = df.iloc[:, :500]
            df1 = pd.concat([df1, temp_df], axis=0)

    # rename the columns in df1 and convert to float
    df1.columns = [i for i in range(df1.shape[1])]
    df1 = df1.rename(columns={0: 'Fault'})
    df1 = df1.astype(float)

    # combine data_pdmp files into df2
    files_pdmp = glob.glob(path + "*data_pdmp*.csv")
    df2 = pd.DataFrame()
    for file in files_pdmp:
        with open(file, 'r') as f:

            csv_string = f.read()

            data = csv_string
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
            df.drop(df.tail(1).index,inplace=True)

            temp_df = df.iloc[:, :500]
            df2 = pd.concat([df2, temp_df], axis=0)

    # rename the columns in df2 and convert to float
    df2.columns = [i for i in range(df2.shape[1])]
    df2 = df2.rename(columns={0: 'Fault'})
    df2 = df2.astype(float)

    # combine data_pin files into df3
    files_pin = glob.glob(path + "*data_pin*.csv")
    df3 = pd.DataFrame()
    for file in files_pin:
        with open(file, 'r') as f:

            csv_string = f.read()

            data = csv_string
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
            df.drop(df.tail(1).index,inplace=True)

            temp_df = df.iloc[:, :500]
            df3 = pd.concat([df3, temp_df], axis=0)

    # rename the columns in df3 and convert to float
    df3.columns = [i for i in range(df3.shape[1])]
    df3 = df3.rename(columns={0: 'Fault'})
    df3 = df3.astype(float)

    return df1, df2, df3


def plot_clusters(df_list, plot_show, num_clusters=11):
    # Create a list of unique colors for each cluster
    colors = sns.color_palette('tab20', num_clusters)

    fig, axs = plt.subplots(1, len(df_list), figsize=(10*len(df_list), 10))

    # Create a separate scatter plot for each dataframe
    for j, df in enumerate(df_list):
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(df[['PC1', 'PC2']])
        df['cluster'] = kmeans.labels_
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





if __name__ == "__main__":
    path = "training_data/"
    df1, df2, df3 = data_preprocessing(path)

    y_true1 = df1["Fault"]
    y_true2 = df2["Fault"]
    y_true3 = df3["Fault"]

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

    # Example usage with three dataframes df1, df2, df3:
    plot_show = False
    plot_clusters([df_pca1, df_pca2, df_pca3], plot_show, num_clusters=11)

    label_cluster_map_1 = pd.crosstab(y_true1, df_pca1["cluster"])
    ami_1 = adjusted_mutual_info_score(y_true1, df_pca1["cluster"])

    label_cluster_map_2 = pd.crosstab(y_true2, df_pca2["cluster"])
    ami_2 = adjusted_mutual_info_score(y_true2, df_pca2["cluster"])

    label_cluster_map_3 = pd.crosstab(y_true3, df_pca3["cluster"])
    ami_3= adjusted_mutual_info_score(y_true3, df_pca3["cluster"])

    print("Adjusted Mutual Information:", ami_1, ami_2, ami_3)