import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo
import pandas as pd
import string
import glob
import sys


def read_murali_csv(fileName):
	
	MEA_data = pd.read_csv(fileName , sep=',', encoding='latin1', skiprows = 6)

	data = {}

	row = 4
	end = 10
	letters = [letter for letter in string.ascii_uppercase]
	letters.remove('I')
	letter = 0  #start with A 

	for j in range(0, 12):
		for i in range(row, end):
			name = str(letters[letter]) + str(i)
			data[name] = []

		if (j < 3): row -= 1
		elif (j > 2) and (j < 8): row += 0
		elif (j > 7): row +=1

		if (j < 3): end += 1
		elif (j > 2) and (j < 8): end += 0
		elif (j > 7): end -=1

		letter += 1

	#print(data)

	MEA_data = MEA_data.reindex(sorted(MEA_data.columns), axis=1)

	MEA_data.rename(columns=lambda x: x.split(' ')[0], inplace=True)

	MEA_data_full = pd.DataFrame.from_dict(data)

	for col in MEA_data_full.columns:
		try:
			MEA_data_full[col] = MEA_data[col]
		except:
			pass

	#print(MEA_data_full)

	MEA_data_full = MEA_data_full.div(1000000)
	
	MEA_data_full = MEA_data_full.reindex(sorted(MEA_data_full.columns), axis=1)

	MEA_data_full.replace(np.nan, -1, inplace=True)

	del MEA_data

	return MEA_data_full

def csv_to_raster(fileName, title, output):

	MEA_data_full = read_murali_csv(fileName)

	spikeTrainArray = []

	t_start = 540
	t_stop = 600

	for col in MEA_data_full.columns:

		values = MEA_data_full[col].values

		values = values[values > t_start]
		values = values[values < t_stop] 

		spikeTrainArray.append(neo.core.SpikeTrain(values * pq.s, t_stop = t_stop * pq.s, t_start = t_start * pq.s))
		#spikeTrainArray.append(MEA_data_full[col].values)

	
	for i, spiketrain in enumerate(spikeTrainArray):
		plt.plot(spiketrain, i * np.ones_like(spiketrain), 'k|', markersize=2)
	
	plt.axis('tight')
	plt.title("Raster Plot - "+title)
	plt.xlim(t_start, t_stop)
	plt.ylim(-5, 125)
	plt.xlabel('Time (s)', fontsize=16)
	plt.ylabel('Channels', fontsize=16)
	plt.gca().tick_params(axis='both', which='major', labelsize=14)
	#plt.show()

	name = output+"\\"+title+"_Raster"+".jpg"
	plt.savefig(name, dpi=600)

	del MEA_data_full
	del spikeTrainArray

	plt.clf()


def main(directory, output):

	directoryPath = directory + "\\*.csv"

	print(directoryPath)

	dirFiles = glob.glob(directoryPath)
	titles = list(map(lambda x: x.split('\\')[5][:-4], dirFiles))

	print(titles)				

	for index, fileName in enumerate(dirFiles):

		print(index, fileName)
		title = titles[index]

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
			#First argument is directory of input .csv files. Second argument is directory to output raster plots
			main("<Default input file directory for .csv spiking data>", "<Default output file directory for raster plots>")
		elif(len(sys.argv) == 2):
			main(sys.argv[1], sys.argv[1])
		else:
			main(sys.argv[1], sys.argv[2])
	except IndexError:
		print("no files in directory")
	except:
		print("something went wrong")

	#main()
