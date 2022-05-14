import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline


data = pd.read_csv('IRIS.csv')
# data = data.drop('Id',axis=1)
data.head()


X = data.iloc[:,0:4]
y = data.iloc[:,-1]
print(X.sample(5))
print(y.sample(5))


data["species"].value_counts()


# use seaborn to make scatter plot showing species for each sample
sns.FacetGrid(data, hue="species", size=4) \
   .map(plt.scatter, "sepal_length", "sepal_width") \
   .add_legend();
sns.FacetGrid(data, hue="species", size=4) \
   .map(plt.scatter, "petal_length", "petal_width") \
   .add_legend();


from sklearn import preprocessing


scaler = preprocessing.StandardScaler()


scaler.fit(X)
X_scaled_array = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)


X_scaled.sample(5)


**K-means clustering**


from sklearn.cluster import KMeans


nclusters = 3
seed = 0


km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(X_scaled)


y_cluster_kmeans = km.predict(X_scaled)
y_cluster_kmeans


from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
score


scores = metrics.silhouette_samples(X_scaled, y_cluster_kmeans)
sns.distplot(scores);


df_scores = pd.DataFrame()
df_scores['SilhouetteScore'] = scores
df_scores['species'] = data['species']
df_scores.hist(by='species', column='SilhouetteScore', range=(0,1.0), bins=20);


**PCA**


from sklearn.decomposition import PCA


ndimensions = 2


pca = PCA(n_components=ndimensions, random_state=seed)
pca.fit(X_scaled)
X_pca_array = pca.transform(X_scaled)
X_pca = pd.DataFrame(X_pca_array, columns=['PC1','PC2']) # PC=principal component
X_pca.sample(5)


y_id_array = pd.Categorical(data['species']).codes
#Categorical.from_array(data['Species']).codes


df_plot = X_pca.copy()
df_plot['ClusterKmeans'] = y_cluster_kmeans
df_plot['SpeciesId'] = y_id_array # also add actual labels so we can use it in later plots
df_plot.head(5)


def plotData(df, groupby):
    "make a scatterplot of the first two principal components of the data, colored by the groupby field"
    fig, ax = plt.subplots(figsize = (7,7))
    cmap = mpl.cm.get_cmap('prism')
    for i, cluster in df.groupby(groupby):
        cluster.plot(ax = ax, 
                     kind = 'scatter', 
                     x = 'PC1', y = 'PC2',
                     color = cmap(i/(nclusters-1)), 
                     label = "%s %i" % (groupby, i), 
                     s=30) 
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title("Principal Components Analysis (PCA) of Iris Dataset");


# plot the clusters each datapoint was assigned to
plotData(df_plot, 'ClusterKmeans')


**GMM**


from sklearn.mixture import GaussianMixture


gmm = GaussianMixture(n_components=nclusters)
gmm.fit(X_scaled)


# predict the cluster for each data point
y_cluster_gmm = gmm.predict(X_scaled)
y_cluster_gmm


# add the GMM clusters to our data table and plot them
df_plot['ClusterGMM'] = y_cluster_gmm
plotData(df_plot, 'ClusterGMM')

