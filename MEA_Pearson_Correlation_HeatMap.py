import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
#import elephant
import pandas as pd
from pprint import pprint
import string

from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import corrcoef

#from elephant.spike_train_correlation import spike_time_tiling_coefficient

from matplotlib import colorbar

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import glob

def CSV_to_matrix(fileName, bin_size):

    MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1')

    spikeTrainArray = []

    t_start = 240.0
    t_stop = 300.0

    for col in MEA_data.columns:
        
        values = MEA_data[col].values
        values = values[values > t_start]
        values = values[values < t_stop]

        spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = t_stop * pq.s, t_start = t_start * pq.s, sort = True))

    cc_matrix = corrcoef(BinnedSpikeTrain(spikeTrainArray, binsize = bin_size*pq.ms))

    return cc_matrix, MEA_data.columns.values

def multiPlot(figures, rows, cols, title, titles, axes = ["", ""]):
    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 600*cols)
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9), showgrid = False)

    outputFile = title+"_pearson_hist.html"

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
    fig.update_xaxes(type='category', tickfont=dict(size=9))
    fig.update_yaxes(type='category', tickfont=dict(size=9))

    #have to update axes titles one by one - reference graph with row and col #

    outputFile = title+"_pearson_heatmap.html"

    fig.write_html(outputFile, auto_open = True)
    #fig.show()

def getCoeffList(matrix):

    coefficientList = []

    for i in range(1, len(matrix)):
        for j in range(0, i):

            if (matrix[i][j] != 0.00):
                coefficientList.append(matrix[i][j])

    #print(coefficientList)

    return coefficientList


def main():
    
    directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"

    figures = []

    figures2 = []


    dirFiles = glob.glob(directoryPath)

    #dirFiles = [dirFiles[3]] + dirFiles[0:3]

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)
    titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
    print(titles)

    title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]
    print(title)

    for index, file in enumerate(dirFiles):

        print(index, file)

        cc_matrix, axis = CSV_to_matrix(file, bin_size=100) 

        #break

        if (index != (len(dirFiles)-1) ):
            figures.append(go.Heatmap(z=cc_matrix, x = axis, y = axis, showscale = False, name = titles[index])) 
        else:
            figures.append(go.Heatmap(z=cc_matrix, x = axis, y = axis, showscale = True, name = titles[index])) 

        x = getCoeffList(cc_matrix)
        figures2.append(go.Histogram(x=x, name = titles[index], xbins = dict(
            start = -0.2,
            end = 1.0), nbinsx = 20, autobinx=False))

        #break

    multiHeatMap(figures, 2, 2, title, titles, ["channels", "channels"])
    multiPlot(figures2, 2, 2, title, titles, ["Pearson Correlation", "count"])

if __name__ == '__main__':
    main()