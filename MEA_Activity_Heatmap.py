import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
import string
import pandas as pd
from pprint import pprint
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from elephant.statistics import instantaneous_rate
from matplotlib import colorbar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from itertools import combinations
from pprint import pprint
import numpy as np
import math

def multiHeatMap(figures, rows, cols, title, titles, axes = ["", ""]):

    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles, vertical_spacing = 0.1, horizontal_spacing = 0.1)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 550*cols)
    fig.update_xaxes(range=[0.5,8.5], tickfont=dict(size=12), dtick=1, visible = False)
    fig.update_yaxes(range=[0.5,8.5], tickfont=dict(size=12), dtick=1, visible = False)

    #have to update axes titles one by one - reference graph with row and col #

    outputFile = title+"_spikerate_heatmap"

    for i in range(2, rows+cols+1):

        fig['layout']['xaxis'+str(i)]['title']= axes[0]
        fig['layout']['yaxis'+str(i)]['title']= axes[1]

    fig.write_html(outputFile+".html", auto_open = True)
    #fig.show()

def CSV_to_matrix(fileName, rows, cols):

    MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1')

    data = {}

    for i in range(1, 9): #column
        for j in range(1, 9):  #row
            name = str(i) + str(j)
            data[name] = []

    #remove corner electrodes that don't exist
    data.pop('11')
    data.pop('18')
    data.pop('81')
    data.pop('88')

    MEA_data_full = pd.DataFrame.from_dict(data)
    MEA_data_full = MEA_data_full.reindex(sorted(MEA_data_full.columns), axis=1)

    spikeTrainArray = []
    firingRateArray = []

    #add one extra to row and column to include the normalization values for heatmap colors
    tiling_matrix = [ [0]*(cols+1) for i in range(0, rows + 1) ]

    t_start = 180.0
    t_stop = 300.0

    for col in MEA_data_full.columns:
        
        try:
            values = MEA_data[col].values
            values = values[values > t_start]
            values = values[values < t_stop]
        except:
            values = []

        firingRate = len(values)/(t_stop - t_start)

        #firing rate, row, column
        firingRateArray.append([firingRate, col[1], col[0]])

    for rate in firingRateArray:

        tiling_matrix[int(rate[1])-1][int(rate[2])-1] = float(rate[0])

    tiling_matrix.reverse()

    return tiling_matrix, firingRateArray


def add_scaler_max(tiling_matrix, rows, cols, maximum):

    tiling_matrix[rows][cols] = 0
    tiling_matrix[rows - 1][cols] = maximum

    return tiling_matrix

def adjust_shape(tiling_matrix, rows, cols):

    tiling_matrix[0][0] = np.nan
    tiling_matrix[7][7] = np.nan
    tiling_matrix[0][7] = np.nan
    tiling_matrix[7][0] = np.nan

    for i in range(0, len(tiling_matrix[cols])):

        tiling_matrix[rows][i] = np.nan

    for i in range(0, len(tiling_matrix[rows])):

        tiling_matrix[i][cols] = np.nan

    for i in range(0, int(len(tiling_matrix)/2)):

        for j in range(0, len(tiling_matrix)):
        
            test = tiling_matrix[(-1*i)+7][j]  
            tiling_matrix[(-1*i)+7][j] = tiling_matrix[i][j]
            tiling_matrix[i][j] = test

    return tiling_matrix

def getCoeffList(matrix):

    coefficientList = []

    for i in range(0, len(matrix)):
        for j in range(0, i):

            coefficientList.append(matrix[i][j])

    return coefficientList

def multiPlot(figures, rows, cols, title, titles, axes = ["", ""], mini = 0, maxi = 100):
    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 600*cols)
    fig.update_xaxes(tickfont=dict(size=12), title_text = axes[0])
    fig.update_yaxes(tickfont=dict(size=12), title_text = axes[1])
    fig.update_xaxes(autorange = False, range = [mini, maxi])
    fig.update_yaxes(autorange = False, range = [0, 1], showgrid = False)

    outputFile = title+"_spikerate"+"_hist"

    fig.write_html(outputFile+".html", auto_open = True)
    #fig.show()

def main():
    
    directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"
    
    dirFiles = glob.glob(directoryPath)

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)
    titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
    title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]

    print(titles)

    figures = []
    figures2 = []

    tilingMatrixArray = []

    maxValues = []

    axis = [i for i in np.arange(1, 8+2, 1)]

    for index, file in enumerate(dirFiles):

        print(index, file)

        tiling_matrix, firingRateArray = CSV_to_matrix(file, 8, 8) 

        tiling_matrix = adjust_shape(tiling_matrix, 8, 8) ##trim to size of mea

        coeffList = getCoeffList(tiling_matrix)

        maxValues.append(np.nanmax(coeffList))

        tilingMatrixArray.append(tiling_matrix)

        figHist = go.Histogram(x=coeffList, name = titles[index], histnorm='probability', autobinx=False, xbins=dict(start='0.0',end=np.nanmax(maxValues), size=1))

        figures2.append(figHist)

    for i in range(0, len(tilingMatrixArray)):

        tilingMatrixArray[i] = add_scaler_max(tilingMatrixArray[i], 8, 8, np.nanmax(maxValues))

        if (i != (len(dirFiles)-1) ):
            figures.append(go.Heatmap(z=tilingMatrixArray[i], x = axis, y = axis, showscale = False, colorscale="rdylbu_r", name = titles[index])) 
        else:
            figures.append(go.Heatmap(z=tilingMatrixArray[i], x = axis, y = axis, showscale = True, colorscale="rdylbu_r", name = titles[index])) 


    multiHeatMap(figures, 2, 2, title, titles, ["", ""])
    multiPlot(figures2, 2, 2, title, titles, ["avg spike rate (Hz)", "probability"], 0, np.nanmax(maxValues))

if __name__ == '__main__':
    main()
