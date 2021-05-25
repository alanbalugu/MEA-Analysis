import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
import pandas as pd
import string
import glob
import sys


def csv_to_raster(fileName, title, output):

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

	t_start = 240.0
	t_stop = 300.0

	for col in MEA_data_full.columns:

		try:
			values = MEA_data[col].values
			values = values[values > t_start]
			values = values[values < t_stop]
		except:
			values = []

		spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = t_stop * pq.s, t_start = t_start * pq.s))
	

	for i, spiketrain in enumerate(spikeTrainArray):
		plt.plot(spiketrain, i * np.ones_like(spiketrain), 'k|', markersize=2)
	
	plt.axis('tight')
	plt.title("Raster Plot - "+title)
	plt.xlim(t_start, t_stop)
	plt.ylim(-1, 60)
	plt.xlabel('Time (s)', fontsize=16)
	plt.ylabel('Channels', fontsize=16)
	plt.gca().tick_params(axis='both', which='major', labelsize=14)
	#plt.show()
	name = output+"\\"+title+"LastMin"+".jpg"
	plt.savefig(name, dpi=600)


	del MEA_data_full
	del spikeTrainArray

	plt.clf()


def main(directory, output):

	directoryPath = directory + "\\*.csv"

	print(directoryPath)

	dirFiles = glob.glob(directoryPath)
	titles = list(map(lambda x: x[len(directoryPath)-5 : len(directoryPath)], dirFiles))
	
	print(titles)		

	for index, fileName in enumerate(dirFiles):

		MEA_num = dirFiles[index][len(directoryPath)+1 : len(directoryPath)+6]

		print(index, fileName)
		title = MEA_num+"_"+titles[index]
		try:
			csv_to_raster(fileName, title, output)
		except:
			print("an error occurred. stopped on: " + fileName)

if __name__ == '__main__':

	#accepts two cmd line arguments, input directory and output directory (no \ at the end of paths)
	print(f"Arguments count: {len(sys.argv)}")
	for i, arg in enumerate(sys.argv):
		print(f"Argument {i:>6}: {arg}")
	try:

		if(len(sys.argv) < 2):
			print("running with default location")
			main("<INSERT DEFAULT DIRECTORY PATH HERE TO .CSV SPIKING DATA FILES>", "<INSERT DEFAULT OUTPUT DIRECTORY PATH HERE")
		elif(len(sys.argv) == 2):
			main(sys.argv[1], sys.argv[1])
		else:
			main(sys.argv[1], sys.argv[2])
	except IndexError:
		print("no files in directory")
	except:
		print("something went wrong")

	#main()
