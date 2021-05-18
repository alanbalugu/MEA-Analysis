import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
#import elephant
import pandas as pd
from pprint import pprint

from elephant.spike_train_correlation import spike_time_tiling_coefficient

from matplotlib import colorbar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import glob

from itertools import combinations

from pprint import pprint
import numpy as np

import math


def CSV_to_matrix(fileName, bin_size):

    MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1')

    MEA_data.replace(0, -1, inplace=True)

    spikeTrainArray = []

    for col in MEA_data.columns:
        
        values = MEA_data[col].values
        values = values[values > 240]
        values = values[values < 300]   

        spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = 300.0 * pq.s, t_start = 240 * pq.s, sort = True))

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
        
def main():
    
    directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"

    dirFiles = glob.glob(directoryPath)

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)
    titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
    title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]

    figures = []

    for index, file in enumerate(glob.glob(directoryPath)):

        print(index, file)

        tiling_matrix, axis = CSV_to_matrix(file, bin_size=50)  #change to 50ms for bin size

        MEArows = list(map(lambda x: int(x[1:2]), axis))
        MEAcols = list(map(lambda x: int(x[0:1]), axis))

        networkData = pd.DataFrame()

        #create a final network dataframe which is essentially the edges
        final_network = pd.DataFrame()

        src = []
        dst = []

        coefficients = []
        distances = []

        for i in range(1, len(tiling_matrix)):
            for j in range(0, i):
                coefficients.append(tiling_matrix[i][j])

                dist = math.sqrt( (MEArows[j] - MEArows[i])**2 + (MEAcols[j] - MEAcols[i])**2 )

                src.append(str(MEArows[i]) + str(MEAcols[i]))
                dst.append(str(MEArows[j]) + str(MEAcols[j]))

                distances.append( dist )

        final_network['Src'] = src
        final_network['Dst'] = dst
        final_network['Corr'] = coefficients
        final_network['Dist'] = distances

        #pprint(final_network)

        outputPath = "<INSERT DIRECTORY PATH HERE TO OUTPUT NETWORK CONNECTIVITY DATA FILE>"

        final_network.to_csv(outputPath+"network_"+title+"_"+titles[index]+".csv",index=False)   #has full connectivity

        #break

if __name__ == '__main__':
    main()
