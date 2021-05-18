import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
#import elephant
import pandas as pd
from pprint import pprint

#from elephant.conversion import BinnedSpikeTrain
#from elephant.spike_train_correlation import corrcoef

from elephant.spike_train_correlation import spike_time_tiling_coefficient

from matplotlib import colorbar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import glob

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn import metrics

from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score

import math

def multiHeatMap(figures, rows, cols, title, titles, axes = ["", ""]):
    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 600*cols)
    fig.update_xaxes(type='category', tickfont=dict(size=9))
    fig.update_yaxes(type='category', tickfont=dict(size=9))

    #have to update axes titles one by one - reference graph with row and col #

    outputFile = title+"_tiling_clustering_heatmap.html"

    fig.write_html(outputFile, auto_open = True)
    #fig.show()

def CSV_to_matrix(fileName, bin_size):

    MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1')

    MEA_data.replace(0, -1, inplace=True)

    spikeTrainArray = []

    for col in MEA_data.columns:
        
        values = MEA_data[col].values
        values = values[values > 0.01]
        values = values[values < 300]   

        spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = 300.0 * pq.s, sort = True))

    #cc_matrix = corrcoef(BinnedSpikeTrain(spikeTrainArray, binsize = bin_size*pq.ms))

    tiling_matrix = [ [0]*len(MEA_data.columns) for i in range(len(MEA_data.columns)) ]

    for i in range(0, len(tiling_matrix)):

        for j in range(0, i+1):
    
            #tiling_matrix[i].append(spike_time_tiling_coefficient(spikeTrainArray[i], spikeTrainArray[j], (bin_size/2)*pq.ms ))
            tiling_matrix[i][j] = spike_time_tiling_coefficient(spikeTrainArray[i], spikeTrainArray[j], (bin_size/2)*pq.ms )

    for i in range(0, len(tiling_matrix)):

        for j in range(i+1, len(tiling_matrix)):

            tiling_matrix[i][j] = tiling_matrix[j][i]

    return tiling_matrix, MEA_data.columns.values
        


# runs DBScan with a certain value of epsilon and min_samples. Returns dataframe with cluster labels, silhouette score, and number of clusters found
def doDBScan(Data, nearness, min_samples):

   new_Data = Data.copy()

   # Creates a clustering model and fits it to the data
   dbscan = DBSCAN(eps=nearness, min_samples=min_samples).fit(new_Data)

   # core samples mask
   core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
   core_samples_mask[dbscan.core_sample_indices_] = True
   labels = dbscan.labels_

   # print(dbscan.labels_.tolist())

   # convert cluster labels to a string list
   labels_list = []
   for each in labels.tolist():
       labels_list.append(str(each))

   # Add the cluster labels to the dataframe
   new_Data['cluster_labels'] = labels_list


   # Number of clusters in labels, ignoring noise points (-1 cluster) if present
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)

   # print the relevant values for the DCScan clustering
   print("eps: ", nearness)
   print('Estimated number of clusters: %d' % n_clusters_)
   print('Estimated number of noise points: %d' % n_noise_)

   silhouette_avg = metrics.silhouette_score(new_Data, labels)

   print("Silhouette Coefficient: %0.3f" % silhouette_avg)

   return new_Data, silhouette_avg, n_clusters_


# runs Kmeans clustering with the dataframe and a given values of k. Returns the dataframe with cluster labels and the silhouette score
def doKMeans(Data, k):
   new_Data = Data.copy()
    
   # do the actual k-means analysis
   kmeans = KMeans(n_clusters=k, random_state = 1)
   cluster_labels = kmeans.fit_predict(new_Data)

   # display clustering accuracy
   silhouette_avg = silhouette_score(new_Data, cluster_labels)
   print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

   new_Data["cluster_labels"] = cluster_labels

   return new_Data, silhouette_avg

# runs heirarchical agglomerative clustering with the dataframe and a given values of k. Returns the dataframe with cluster labels and the silhouette score.
def doHierarchical(Data, k):

   new_Data = Data

   #do hierarchical clustering and fit to data
   hierarchical = AgglomerativeClustering(n_clusters = k)
   cluster_labels = hierarchical.fit_predict(new_Data)
   
   silhouette_avg = silhouette_score(new_Data, cluster_labels)
   print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

   new_Data["cluster_labels"] = cluster_labels

   return new_Data, silhouette_avg

#runs clustering for a dataframe through a range of values (k for kmeans, eps for dbscan). Returns dataframe with best clustering cluster labels (max silhouette score) 
def cycleClustering(Data, type_clus, range_list, silh_score_list, cluster_vals, min_samples = 5):

    max_silh_score = 0
    best_cluster = 1

    normalized_Data = Data.copy()

    #for K means or hierarchical clustering
    if (type_clus == "K" or type_clus == "H"):

        #runs clustering on all values of k in range
        for i in range_list:

            silh_score = 0

            if (type_clus == "K"):
                clustered_data, silh_score = doKMeans(normalized_Data, i)
                clustered_data, silh_score = doKMeans(normalized_Data, i)
            else:
                clustered_data, silh_score = doHierarchical(normalized_Data, i)
                clustered_data, silh_score = doHierarchical(normalized_Data, i)

            cluster_vals.append(i)
            silh_score_list.append(silh_score)

            #save the k values for the 'best' clustering
            if (silh_score > max_silh_score):
                max_silh_score = silh_score
                best_cluster = i

        #do a final clustering with the best parameter
        if (type_clus == "K"):
                clustered_data, silh_score = doKMeans(normalized_Data, best_cluster)
        else:
                clustered_data, silh_score = doHierarchical(normalized_Data, best_cluster)
    
    #for dbscan clustering
    else:

        best_cluster = 0.0
        best_eps = 0.1

        #runs dbscan on all epsilon values in the range
        for i in range_list:
            try:
                clustered_data, silh_score, num_clusters = doDBScan(normalized_Data, i, min_samples)
                cluster_vals.append(i)
                silh_score_list.append(silh_score)

                if (num_clusters > best_cluster):
                    best_cluster = num_clusters
                
                #save the k values for the 'best' clustering
                if (silh_score > max_silh_score):
                    max_silh_score = silh_score
                    best_eps = i

                clustered_data = pd.DataFrame()
                    
            except:
                print("didn't work")

        #do a final clustering with the best parameter

        try:
            clustered_data, silh_score, num_clusters = doDBScan(Data, best_eps, min_samples)
        except:
            clustered_data = Data
            silh_score = 0
            num_clusters = 0


    print("best clusters: ", best_cluster)
    print("max silh. score: ", max_silh_score)

    return clustered_data, silh_score_list, cluster_vals

#creates a scatter plot and color codes the values by cluster labels if that parameter is passed in. Adds annotations to each data point as well.
def scatterPlot3(X_Data, Y_Data, x_axis, y_axis, title, save, clusterLabels = None):
    
    plt.figure(1, figsize = (6,8))

    ax = plt.axes()


    if (clusterLabels != None):
        plt.scatter(X_Data, Y_Data, s=20, cmap = 'rainbow', c=clusterLabels)
    else:
        plt.scatter(X_Data, Y_Data, s=20)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    ax.tick_params(which="both", bottom=False, left=False, labelsize = 7)
    #plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    if (save == True):
        plt.savefig(title + '.png')
        plt.clf()
    else:
        plt.show()
        plt.clf()

def main():
    
    directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"

    dirFiles = glob.glob(directoryPath)

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)
    titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
    title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]

    clusterData = pd.DataFrame()

    figures = []

    for index, file in enumerate(glob.glob(directoryPath)):
        coefficients = []
        x_pos1 = []
        y_pos1 = []
        x_pos2 = []
        y_pos2 = []
        distances = []


        clusterData['Coefficients'] = []
        #clusterData['Distances'] = []

        #add in electrode coordinates for clustering
        #---------------------------------
        # clusterData['x_1'] = []
        # clusterData['y_1'] = []
        # clusterData['x_2'] = []
        # clusterData['y_2'] = []
        # ---------------------------------

        tiling_matrix, axis = CSV_to_matrix(file, bin_size=50)  #change to 50ms for bin size

        MEArows = list(map(lambda x: int(x[0:1]), axis))
        MEAcols = list(map(lambda x: int(x[1:2]), axis))

        for i in range(1, len(tiling_matrix)):
            for j in range(0, i):
                coefficients.append(tiling_matrix[i][j])

                #--------------------------------
                x_pos1.append( MEArows[i] )
                y_pos1.append( MEAcols[i] )
                x_pos2.append( MEArows[j] )
                y_pos2.append( MEAcols[j] )

                #print( (MEArows[j] - MEArows[i])**2 + (MEAcols[j] - MEAcols[i])**2 )
                dist = math.sqrt( (MEArows[j] - MEArows[i])**2 + (MEAcols[j] - MEAcols[i])**2 )
                distances.append( dist )
                #_-------------------------------

        clusterData['Coefficients'] = coefficients
        #clusterData['Distances'] = distances

        # add in electrode coordinates for clustering
        # ------------------------------------------
        # clusterData['x_1'] = x_pos1
        # clusterData['y_1'] = y_pos1
        # clusterData['x_2'] = x_pos2
        # clusterData['y_2'] = y_pos2
        # ------------------------------------------

        silh_score_list = []
        cluster_vals = []

        #cycle clustering to find best cluster size
        #clustered_data, silh_score_list, cluster_vals = cycleClustering(clusterData, "D", np.arange(0.2, 7, 0.5), silh_score_list, cluster_vals, min_samples=15)
        #clustered_data, silh_score_list, cluster_vals = cycleClustering(clusterData, "K", np.arange(2,10,1), silh_score_list, cluster_vals)

        newData, silh_score = doKMeans(clusterData, 5)
        #newData, silh_score = doKMeans(clusterData, 6)
        #newData = clustered_data

        #clustering plots
        #scatterPlot3(cluster_vals, silh_score_list, "cluster size", "silhouette score", "cycle clustering "+title+" "+titles[index], False)

        scatterPlot3(newData['cluster_labels'], newData['Coefficients'], "cluster labels", "tiling coefficient", "clusters = 5", False)
        #scatterPlot3(newData['cluster_labels'], newData['Distances'], "cluster labels", "distances", "clusters = 5", False)

        # add in electrode coordinates to heatmap (not included in clustering)
        #------------------------------------------
        newData['x_1'] = x_pos1
        newData['y_1'] = y_pos1
        newData['x_2'] = x_pos2
        newData['y_2'] = y_pos2
        #-----------------------------------------

        cluster_matrix = [ [0]*len(axis) for i in range(len(axis)) ]

        for i, i1 in enumerate(axis):
            for j, j1 in enumerate(axis):

                try:
                    cluster_matrix[i][j] = newData.loc[ (newData['x_1'] == int(str(j1)[0:1])) & (newData['y_1'] == int(str(j1)[1:2])) 
                    & (newData['x_2'] == int(str(i1)[0:1])) & (newData['y_2'] == int(str(i1)[1:2]))]['cluster_labels'].values[0] + 1
                except:
                    cluster_matrix[i][j] = 0

        #print(cluster_matrix)

        figures.append(go.Heatmap(z=cluster_matrix, x = axis, y = axis))

        networkData = pd.DataFrame()

        clusterData = pd.DataFrame()

    multiHeatMap(figures, 2, 2, title, titles, ["channels", "channels"])

if __name__ == '__main__':
    main()
