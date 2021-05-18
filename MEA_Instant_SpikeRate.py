import matplotlib.pyplot as plt
import elephant.conversion as conv
import elephant.spike_train_generation
import quantities as pq
import numpy as np
import elephant.cell_assembly_detection as cad
import neo
import pandas as pd
from pprint import pprint
import string
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from matplotlib import colorbar
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob


def CSV_to_bar(fileName, name, bin_size):

	MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1')
	MEA_data.replace(0, -1, inplace=True)

	spikeTrainArray = []
	binnedSpikeTrainArray = []

	t_start = 240.0
	t_stop = 300.0

	for col in MEA_data.columns:
		
		values = MEA_data[col].values
		values = values[values > t_start]
		values = values[values < t_stop]

		spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = t_stop * pq.s, t_start = t_start * pq.s, sort = True))

	binnedSpikeTrainArray = conv.BinnedSpikeTrain(spikeTrainArray, binsize= bin_size*pq.ms, t_start = t_start * pq.s, t_stop = t_stop * pq.s).to_array()

	#print(list(binnedSpikeTrainArray[3]))

	finalArray = [0 for i in binnedSpikeTrainArray[0]]
	indexArray = [(t_start+index*bin_size/1000) for index,i in enumerate(binnedSpikeTrainArray[0])]

	for index, array in enumerate(binnedSpikeTrainArray):

		for index2, item in enumerate(array):

			finalArray[index2] += item  #*(1000/bin_size)
	
	#print(finalArray)

	del spikeTrainArray
	del binnedSpikeTrainArray

	return finalArray, indexArray

	
def multiPlot(figures, rows, cols, title, titles, bins, axes = ["", ""]):
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
	#fig.update_xaxes(autorange = False, range = [-1, bins])
	#fig.update_yaxes(range = [0, 5000])


	outputFile = title+"_instant_spikeRate"+".html"

	fig.write_html(outputFile, auto_open = True)
	#fig.show()

def getCoeffList(matrix):

	coefficientList = []

	for i in range(1, len(matrix)):
		for j in range(0, i):

			if (matrix[i][j] != 0.00):
				coefficientList.append(matrix[i][j])

	return coefficientList


def main():
	
	directoryPath = "<INSERT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>"

	dirFiles = glob.glob(directoryPath)

	titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
	title = dirFiles[0][len(directoryPath)+1 : len(directoryPath)+6]

	figures = []

	print(titles)		

	bin_size = 300

	for index, fileName in enumerate(dirFiles):

		print(index, fileName)
		yArr, xArr = CSV_to_bar(fileName, titles[index], bin_size)

		fig = go.Bar(x=xArr, y=yArr, marker_color='indianred')

		figures.append(fig)

	multiPlot(figures, 2, 2, title, titles, 60*(1000/bin_size), ["time bins", "spikes"])
		

if __name__ == '__main__':
	main()