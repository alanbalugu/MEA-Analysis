import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
import string
import pandas as pd
from pprint import pprint
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from elephant.statistics import isi
from matplotlib import colorbar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from itertools import combinations
from pprint import pprint
import numpy as np
import math

def CSV_to_list(fileName):

    MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1')

    MEA_data.replace(0, -1, inplace=True)

    spikeTrainArray = []
    isiArray = []

    t_start = 240
    t_stop = 300

    for col in MEA_data.columns:
        
        values = MEA_data[col].values
        values = values[values > t_start]
        values = values[values < t_stop] 

        #spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = t_stop * pq.s, t_start = t_start * pq.s))
        isiArr = isi(neo.core.SpikeTrain(values * pq.s, t_stop = t_stop * pq.s, t_start = t_start * pq.s)).rescale(pq.ms)
        
        for val in isiArr:

            if(val > 0):
                isiArray.append( math.log10(val.item()) )


    return isiArray


def multiPlot(figures, rows, cols, title, titles, axes = ["", ""], mini = 0, maxi = 100):

    fig = make_subplots(rows = rows, cols = cols, subplot_titles = titles, vertical_spacing = 0.1, horizontal_spacing = 0.1)

    index = 0

    for i in range(1, rows+1):
        for j in range(1, cols+1):

            if (index < len(figures)):
                fig.add_trace(figures[index], row = i, col = j)
                index += 1

    fig.update_layout(title_text = title, xaxis_title_text = axes[0], yaxis_title_text = axes[1], width = 600*rows, height = 550*cols)
    fig.update_xaxes(tickfont=dict(size=12), title_text = axes[0])
    fig.update_yaxes(tickfont=dict(size=12), title_text = axes[1])
    #fig.update_yaxes(autorange = False, range = [0, 1], showgrid = False)

    #fig.update_yaxes(range = [0, 10000])
    fig.update_xaxes(autorange = False, range = [mini, maxi])

    outputFile = title+"_isi_hist"

    fig.write_html(outputFile+".html", auto_open = True)
    #fig.show()

def main():
    
    directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"
    
    dirFiles = glob.glob(directoryPath)

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)
    titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))

    title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]
    
    print(titles)

    figures2 = []

    axis = [i for i in np.arange(1, 12+2, 1)]

    for index, file in enumerate(dirFiles):

        print(index, file)

        isiArray= CSV_to_list(file)  #change to 50ms for bin size

        #print(isiArray)

        figHist = go.Histogram(x=isiArray, name = titles[index], histnorm = 'probability', autobinx=False, xbins=dict(start=0,end=5, size=0.2))
        #figHist = go.Histogram(x=isiArray, name = titles[index])

        figures2.append(figHist)

    multiPlot(figures2, 2, 2, title, titles, ["Log(Inter Spike Interval) (ms)", "Probability"], 1, 5)


if __name__ == '__main__':
    main()
