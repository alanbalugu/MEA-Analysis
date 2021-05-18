#network of nodes
#Alan Balu

#import statements
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd

from pprint import pprint

import networkx as nx
import matplotlib.pyplot as plt
import community

import glob

import statistics

#function to examine the degree of nodes in the network and generate plots to see this
def degreeAnalysis(merged_graph, title, save = False):

	#get the sequence of degree for each node
	degree_sequence = [d for n, d in merged_graph.degree()] # degree sequence
	print(degree_sequence)

	#plot the distribution for nodes
	plt.hist(degree_sequence, bins = 15)
	plt.title("Degree Distribution")
	plt.ylabel('Count')
	plt.xlabel('Degree of Nodes')

	if (save): plt.savefig(title + '_degree_distr.png')
	if (not save): plt.show()

	plt.clf()

	#plot the degree of each state as a scatter plot
	plt.scatter(list(merged_graph.nodes), degree_sequence)
	plt.title(title+" Degree of Each Node")
	ax = plt.axes()
	ax.tick_params(which="both", bottom=False, left=True, labelsize = 7)
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
	plt.ylabel('Degree')
	plt.xlabel('Nodes')

	if (save): plt.savefig(title + '_node_degrees.png')
	if (not save): plt.show()
	plt.clf()

#function to complete general analysis of network and its connectivity and print those values to the console
def generalAnalysis(merged_graph):

	#print the density of the graph
	print("density of graph: ", nx.density(merged_graph))

	# Compute and print number of nodes and edges
	nbr_nodes = len(merged_graph.nodes())
	nbr_edges = len(merged_graph.edges())

	print("Number of nodes:", nbr_nodes)
	print("Number of edges:", nbr_edges)

	#-------------------------------------------------
	#determine the number of connected components
	nbr_components = nx.number_connected_components(merged_graph)
	print("Number of connected components:", nbr_components)

	#------------------------------------------------------

	#determine the number of triangles
	print("all triangles: \n")
	print(nx.triangles(merged_graph).values())

	triangles = 0
	for value in nx.triangles(merged_graph).values():
		triangles += value

	print("number of triangles total: ", triangles/3)

	#------------------------------------------------------

	#determine the largest connected component and its properties
	test = [merged_graph.subgraph(c) for c in nx.connected_components(merged_graph)]
	maximum = max(len(c.nodes()) for c in test)

	print("largest connected component: ")
	print("number of nodes: ", maximum)

	#determine which connected component is the largest and print that
	max_len = 0
	largest_CC = []
	for item in test:
		if (len(item.nodes()) > max_len):
			max_len = len(item.nodes())
			largest_CC = item.nodes()

	print(largest_CC)

#function to examine the centralities of the network (betweenness and degree)
def centralities(merged_graph, title, posDict, save = False):

	#calculate degree centrality for all nodes
	centralities = nx.degree_centrality(merged_graph)
	values = [centralities[node] for node in merged_graph.nodes()]

	print("degree centrality average: ", sum(values)/len(values))

	#plot graph with degree centrality as colors
	plt.clf()
	plt.title(title+" Degree Centrality")
	#my_pos = nx.spring_layout(merged_graph, seed = 100)
	nx.draw(merged_graph, pos = posDict, cmap = plt.get_cmap('coolwarm'), font_color = 'black', node_color = values, node_size=90, with_labels=True, font_size = 6)
	
	if (save): plt.savefig(title + '_degree_centrality.png')
	if (not save): plt.show()
	plt.clf()

	#calculate betweenness centrality for all nodes
	centralities = nx.betweenness_centrality(merged_graph)
	values = [centralities[node] for node in merged_graph.nodes()]

	print("betweenness centrality average: ", sum(values)/len(values))

	#plot graph with betweenness centrality as colors
	plt.clf()
	plt.title(title+" Betweenness Centrality")
	nx.draw(merged_graph, pos = posDict, cmap = plt.get_cmap('coolwarm'), font_color = 'black', node_color = values, node_size=90, with_labels=True, font_size = 6)
	
	if (save): plt.savefig(title + '_between_centrality.png')
	if (not save): plt.show()
	plt.clf()

#function to partition the network and examine the partitioning through plots and statistics
def partitioning(merged_graph, posDict = {}, weights = [], title = "", save = False):

	#partition the network in the best way
	partition = community.best_partition(merged_graph)

	# Print clusters and its characteristics
	print()
	print("Clusters")
	print(partition)

	things = [partition.get(item) for item in partition]
	print("number of clusters: ", max(things)+1)
	print("average clustering coefficient: ", nx.average_clustering(merged_graph))

	# Get the values for the clusters and select the node color based on the cluster value
	values = [partition.get(node) for node in merged_graph.nodes()]
	print(values)

	plt.title("Network Partitioning")
	nx.draw(merged_graph, posDict, cmap = plt.get_cmap('viridis'), node_color = values, node_size=300, with_labels=True, font_size = 9, font_color = 'white', edge_color = 'g', width = weights)
	
	if (save): plt.savefig(title + '_partitioning.png')
	if (not save): plt.show()

	plt.clf()

	# Determine the final modularity value of the network
	modValue = community.modularity(partition, merged_graph)
	print("modularity: {}".format(modValue))

	#print("small-worldness: {}".format( nx.sigma(merged_graph) ))

	degree_sequence = [d for n, d in merged_graph.degree()] # degree sequence

	#plot the degree of each state as a scatter plot and color it by the partitioning labels
	plt.scatter(list(merged_graph.nodes), degree_sequence, cmap = 'rainbow', c=values)
	ax = plt.axes()
	ax.tick_params(which="both", bottom=False, left=True, labelsize = 7)
	plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
	plt.title(title + " Degree Distribution with Colors as Partitioning Labels")
	plt.ylabel('Degree')
	plt.xlabel('Nodes')

	if (save): plt.savefig(title + '_degree_distr_labeled.png')
	if (not save): plt.show()

	plt.clf()

#driver program to analyze the network and create visualizations
def main():
	print("main")

	directoryPath = "<INSERT DIRECTORY PATH HERE FOR NETWORK CONNECTIVITY DATA FILES>"

	dirFiles = glob.glob(directoryPath)

    #number of characters before the "div..." part in the full PYTHON filename (slash = double slash)

	for index, file in enumerate(dirFiles):

		title = dirFiles[index][len(directoryPath)+3 : len(directoryPath)+14]

		#create an empty graph
		merged_graph = nx.Graph()

		pos = []

		print(index, file)

		sources = []
		targets = []
		weights = []
		distances = []

		with open(file, 'r') as f:
			junk = f.readline()
			for line in f:
				data = line.strip().split(',')
				#print(data)
				source = int(data[0][1]+data[0][0]) #13 -> col, row
				sources.append(source)
				
				target = int(data[1][1]+data[1][0])
				targets.append(target)
				
				try:
					weight = float(data[2]) #+ 0.5/float(data[3])
				except:
					weight = 0

				weights.append(weight)

				dist = float(data[3])
				distances.append(dist)

				#if (weight > 0.5):      #0.6 works well for clustering, but higher is better for modularity score and seeing stronger relationships
					#merged_graph.add_edge(source, target, weight=weight)

		avg_weight = statistics.mean(weights)
		stdv_weight = statistics.stdev(weights)

		for i in range(0, len(sources)):
			if (weights[i] > (avg_weight + 1* stdv_weight)): 

				merged_graph.add_edge(sources[i], targets[i], weight=weights[i], length = distances[i])

		for node in merged_graph.nodes:
			pos.append( ( int(str(node)[0:1]), (int(str(node)[1:2])*-1.0)+9 ) )

		posDict = {}

		for index, node in enumerate(merged_graph.nodes):
			posDict[node] = pos[index]

		#print(merged_graph.nodes)
		#print(posDict)

	    #visualize the network
		#nx.draw_kamada_kawai(merged_graph, node_size=9, edge_size = 1, node_color='b', with_labels = True, font_color = "purple")
		plt.title(title+" Channel Connectivity")
		nx.draw(merged_graph, posDict, node_size=300, node_color='b', with_labels = True, font_color = "white", edge_color = 'g', width = weights)
		
		plt.show()
		#plt.savefig(title + '_network.png')

		plt.clf()

		#'''
		#degreeAnalysis(merged_graph, title, False)
		generalAnalysis(merged_graph)
		#centralities(merged_graph, title, posDict, False)
		partitioning(merged_graph, posDict, weights, title, False)
		#'''

if __name__ == '__main__':
	main()
