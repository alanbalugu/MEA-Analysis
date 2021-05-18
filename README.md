# MEA-Analysis
Python scripts for various analyses of MEA spiking data. Includes raster plots, ISI histograms, Spike rate plots, STTC plots, and graph-based analysis.

Files are run as usual for Python. 
MEA-RasterPlots.py can be run with optional command line arguments: First argument is the input file directory path (ex. "/Users/Me/DataFiles"). Second argument is formatted similarly and is the output directory for the raster plot images.

Most files include a start and stop time in seconds (t_start and t_stop) in the code that determines what portion of the recording to read the spiking data from. (ex. 240s to 300s) . -> change in script to adjust to needs.

Output file name and such as determined by extracting a portion of the file name to be used as the title for each file. This will need to be altered as needed in each script to correctly parse the directory and filenames of the .csv spiking data files. 

  In MEA_RasterPlots.py:
  - ex. "titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))" -> this line needs to be changed, as it assumed that the last 5 characters in the data file's name is the MEA number, which is used in the file names for output. Input file names were formatted as: ".../DIV18_33211_E2.csv"
  - ex. "MEA_num = dirFiles[index][len(directoryPath)+1 : len(directoryPath)+6]" -> this also needs to be changed, as it assumes that the Day in Vitro (DIV) number is encoded as such in the file name. Files names were formatted as: ".../DIV18_33211_E2.csv"

MEA_Clustering.py is the most complicated script that has many lines commented out. The script can cycle through data files and cluster the data with different values for k in KMeans or Hierarchical.

  For cycle clustering for KMeans, as an example, modify the script by uncommenting lines to be like:

    #cycle clustering to find best cluster size
    #clustered_data, silh_score_list, cluster_vals = cycleClustering(clusterData, "D", np.arange(0.2, 7, 0.5), silh_score_list, cluster_vals, min_samples=15)
    clustered_data, silh_score_list, cluster_vals = cycleClustering(clusterData, "K", np.arange(2,10,1), silh_score_list, cluster_vals)

    #newData, silh_score = doKMeans(clusterData, 5)
    #newData, silh_score = doKMeans(clusterData, 6)
    #newData = clustered_data

    #clustering plots
    scatterPlot3(cluster_vals, silh_score_list, "cluster size", "silhouette score", "cycle clustering "+title+" "+titles[index], False)

    #scatterPlot3(newData['cluster_labels'], newData['Coefficients'], "cluster labels", "tiling coefficient", "clusters = 5", False)
    #scatterPlot3(newData['cluster_labels'], newData['Distances'], "cluster labels", "distances", "clusters = 5", False)
    
  For KMeans clustering with a particular size cluster (5 in this example), modify the script as:
  
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
