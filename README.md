# MEA-Analysis
Python scripts for various analyses of MEA spiking data. Includes raster plots, ISI histograms, Spike rate plots, STTC plots, and graph-based analysis.

**Input files are encoded with columns for each electrode and the spike times populating the columns. Reading in such files are customized for the MCS MEA2100 system with 60 electrodes arranged in an 8x8 grid. Reading in data from other sources will require modification to the "csv_to_raster() or CSV_to_matrix() functions to properly read in the .csv spiking data file, as well as modification to the heatmap scripts and network creation/analysis scripts (they use the electrode name for the position in the grid)**

Files are run as usual for Python. 
  MEA-RasterPlots.py can be run with optional command line arguments: First argument is the input file directory path (ex. "/Users/Me/DataFiles"). Second argument is formatted similarly and is the output directory for the raster plot images. 
  Other scripts require the file name hardcoded into the script (ex. "/Users/Me/DataFiles/*.csv")

Reading in files requires a mapping of the electrodes found in the datafile to all electrodes possible, so that all electrodes, even if silent, are present in the DataFrame. Modify the csv_to_raster() and CSV_to_matrix() methods as needed to accomplish this. 

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

Contact agb76 @ georgetown.edu for questions/concerns. These scripts were made for the Dzakpasu Lab in the Georgetown University Department of Pharmacology and Physiology. The accuracy of the plots are not guaranteed. Use with your own discretion.
