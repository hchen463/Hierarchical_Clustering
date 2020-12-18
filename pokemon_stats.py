"""
Implement hierarchical clustering to the Pokemon data.

@author: Hongxu Chen
"""

import csv
import math
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage



def load_data(filepath):
    """ Load the first 20 pokemons
    
    Args:
        filepath: the path of the file
        
    Return: 
        data (list): list of dictionary containing the first 20 pokemons stats.
    
    """
    
    data = []
    # Open the pokemon file and store the corresponding data
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        index = 0
        for row in reader:
            dic = {}
            dic['Name'] = row['Name']
            dic['Type 1'] = row['Type 1']
            dic['Type 2'] = row['Type 2']
            dic['Total'] = row['Total']
            dic['HP'] = row['HP']
            dic['Attack'] = row['Attack']
            dic['Defense'] = row['Defense']
            dic['Sp. Atk'] = row['Sp. Atk']
            dic['Sp. Def'] = row['Sp. Def']
            dic['Speed'] = row['Speed']
            data.append(dic)
            index = index + 1
            if index == 20:
                break
    return data



def calculate_x_y(stats): 
    """ Compute the x and y stats of a specific pokemon
    
    Args:
        stats (dictionary): a dictionary of the stats of a pokemon
        
    Return: 
        feature (tuple): feature value of the pokemon represented as (x,y),
        x denotes the attack value, y denotes the defense value.
    
    """
    x = int(stats['Attack']) + int(stats['Sp. Atk']) + int(stats['Speed'])
    y = int(stats['Defense']) + int(stats['Sp. Def']) + int(stats['HP'])
    feature = (x,y)
    return feature


raw_data = load_data('Pokemon.csv')
data = [None]*len(raw_data)
for i in range(len(data)):
    data[i] = calculate_x_y(raw_data[i])





def hac(dataset):
    """ Apply the HAC to the dataset
    
    Args:
        dataset (list): A collection of m observation vectors in n dimensions may be passed as an m by n array 
        (for us, this will be a list of tuples, not a numpy array like for linkage()!). 
        All elements of the condensed distance matrix must be finite, i.e. no NaNs or infs. 
        In our case, m is the number of Pokemon (here 20) and n is 2: the x and y features for each Pokemon.
        
    Return: 
        hac (NumPy matrix):  An (m-1) by 4 matrix Z. At the i-th iteration, 
        clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster m + i. 
        A cluster with an index less than m corresponds to one of the m original observations. 
        The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] 
        represents the number of original observations in the newly formed cluster.
    
    """
    m = len(dataset)
    # Build a forest to store all the clusters
    forest = {}
    length = 0
    # Initialize the forest, each vector is a cluster
    for i in range(m):
        # Filter all the NaN and inf vector
        if not math.isnan(dataset[i][0]) and not math.isinf(dataset[i][0]) \
            and not math.isnan(dataset[i][1]) and not math.isinf(dataset[i][1]):
            forest[i] = dataset[i]
            length = length + 1
    index = m
    hac = []
    # Start the hac algorithm
    for i in range(length-1):
        mini = float('inf')
        # Find the minimum distance and the corresponding two clusters
        for x in forest:
            for y in forest:
                if x != y:
                    distance = dist(forest[x], forest[y])
                    if distance < mini:
                        mini = distance
                        combine = (x, y)
                    # Implement the tie breaking
                    elif distance == mini:
                        if x < combine[0]:
                            combine = (x, y)
                        elif x == combine[0] and y < combine[1]:
                            combine = (x, y)
        x = combine[0]
        y = combine[1]
        forest[index] = []
        # Check whether the cluster has 1 or more elements and then add the 
        # new cluster to the forest.
        if type(forest[x]) == list:
            forest[index] = forest[index] + forest[x]
        else:
            forest[index].append(forest[x])
        if type(forest[y]) == list:
            forest[index] = forest[index] + forest[y]
        else:
            forest[index].append(forest[y])
        # Remove the two old clusters from the forest.
        forest.pop(x)
        forest.pop(y)
        # Append the new cluster into the hac matrix.
        cluster = [None]*4
        cluster[0] = min(x,y)
        cluster[1] = max(x,y)
        cluster[2] = mini
        cluster[3] = len(forest[index])
        index = index+1
        hac.append(cluster)
    return np.array(hac)
    




def dist(set1, set2):
    """ Compute the distance between two clusters
    
    Args:
        set1 (list or tuple): a cluster
        set2 (list or tuple): a cluster
        
    Return: 
        mini (float): minimum distance between two clusters.
    
    """
    # Check whether the clusters are list or tuple.
    if not type(set1) == list and not type(set2) == list:
        return math.sqrt(math.pow(set1[0] - set2[0], 2) + math.pow(set1[1] - set2[1], 2))

    elif type(set1) == list and not type(set2) == list:
        mini = math.sqrt(math.pow(set1[0][0] - set2[0], 2) + math.pow(set1[0][1] - set2[1], 2))
        for i in range(1, len(set1)):
            dist = math.sqrt(math.pow(set1[i][0] - set2[0], 2) + math.pow(set1[i][1] - set2[1], 2))
            if dist < mini:
                mini = dist
        return mini
    
    elif not type(set1) == list and type(set2) == list:
        mini = math.sqrt(math.pow(set1[0] - set2[0][0], 2) + math.pow(set1[1] - set2[0][1], 2))
        for i in range(1, len(set1)):
            dist = math.sqrt(math.pow(set1[0] - set2[i][0], 2) + math.pow(set1[1] - set2[i][1], 2))
            if dist < mini:
                mini = dist
        return mini
    
    elif type(set1) == list and type(set2) == list:
        mini = math.sqrt(math.pow(set1[0][0] - set2[0][0], 2) + math.pow(set1[0][1] - set2[0][1], 2))
        for i in range(0, len(set1)):
            for j in range(0, len(set2)):
                dist = math.sqrt(math.pow(set1[i][0] - set2[j][0], 2) + math.pow(set1[i][1] - set2[j][1], 2))
                if dist < mini:
                    mini = dist
        return mini
    
    

    
    
    
    
    
    