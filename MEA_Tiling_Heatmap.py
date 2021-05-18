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

    avgTiling = 0.0
    count = 0;

    for i in range(0, len(tiling_matrix)):

        for j in range(0, i+1):
    
            #tiling_matrix[i].append(spike_time_tiling_coefficient(spikeTrainArray[i], spikeTrainArray[j], (bin_size/2)*pq.ms ))
            tiling_matrix[i][j] = spike_time_tiling_coefficient(spikeTrainArray[i], spikeTrainArray[j], (bin_size/2)*pq.ms )
            
            if(not np.isnan(tiling_matrix[i][j])):
                avgTiling += float(tiling_matrix[i][j])
                count += 1

    for i in range(0, len(tiling_matrix)):

        for j in range(i+1, len(tiling_matrix)):

            tiling_matrix[i][j] = tiling_matrix[j][i]

    print("\taverage tiling coeff: " + str(avgTiling/count) )

    return tiling_matrix, MEA_data.columns.values

def multiPlot(figures, rows, cols, title, titles, axes = ["", ""]):
    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 600*cols)
    fig.update_xaxes(tickfont=dict(size=12), title_text = axes[0])
    fig.update_yaxes(tickfont=dict(size=12), title_text = axes[1], showgrid = False)
    fig.update_xaxes(autorange = False, range = [0,1])

    outputFile = title+"_tiling_hist.html"

    fig.write_html(outputFile, auto_open = True)
    #fig.show()

def multiHeatMap(figures, rows, cols, title, titles, axes = ["", ""]):
    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 600*cols)
    fig.update_xaxes(type='category',tickfont=dict(size=12), title_text = axes[0])
    fig.update_yaxes(type='category',tickfont=dict(size=12), title_text = axes[1])

    #have to update axes titles one by one - reference graph with row and col #

    outputFile = title+"_tiling_heatmap.html"

    fig.write_html(outputFile, auto_open = True)
    #fig.show()

def getCoeffList(matrix):

    coefficientList = []

    for i in range(1, len(matrix)):
        for j in range(0, i):
            coefficientList.append(matrix[i][j])

    #print(coefficientList)

    return coefficientList


def main():
    
    directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"

    figures = []
    figures2 = []

    dirFiles = glob.glob(directoryPath)

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)
    titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
    title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]

    for index, file in enumerate(dirFiles):

        print(index, file)

        tiling_matrix, axis = CSV_to_matrix(file, bin_size=50)  #bin size is really value*2!

        if (index != (len(dirFiles)-1) ):
            figures.append(go.Heatmap(z=tiling_matrix, x = axis, y = axis, showscale = False, name = titles[index])) 
        else:
            figures.append(go.Heatmap(z=tiling_matrix, x = axis, y = axis, showscale = True, name = titles[index])) 

        x = getCoeffList(tiling_matrix)
        figures2.append(go.Histogram(x=x, name = titles[index], histnorm = 'probability', xbins=dict(start=0,end=1, size=0.05)))

        #break

    #multiHeatMap(figures, 2, 2, title, titles, ["electrode", "electrode"])
    #multiPlot(figures2, 2, 2, title, titles, ["tiling coefficient", "probability"])

if __name__ == '__main__':
    main()