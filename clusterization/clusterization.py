import numpy as np
import matplotlib.pyplot as plt
from typing import (Final, List, Dict)

def distance(*args: float) -> float:
    """ Decart distance """
    return np.sqrt(np.sum(np.square(args)))

def neighborhood( *coords: np.ndarray, radius: float) -> List[int]:
    """ 
    Return list of neighboring beads 
    
    input parametrs
    radius - minimal distance between neighboring beads
    *coords - coordinates of beads
    
    output parametrs
    bonds: List[int]
    """
    bonds = []  
    if len(set([len(coord) for coord in coords])) != 1:
        raise Exception('Different dimension of input coordinates')

    num_beads = len(coords[0]) 
    for i in range(num_beads-1):
        for j in range(i+1, num_beads):
            dr = [coords[dim][i]-coords[dim][j] for dim in range(len(coords))]
            if distance(dr) < radius:
                bonds.append((i,j))
    return bonds

def stratification(bonds: List, number_of_beads) -> Dict: 
    """ Define claster's id 
    input parametrs
    number_of_beads - number of all beads
    bonds - list of beads within a radius from each other 
    
    output param
    Dict  {cluster : list(bead_1, .., bead_n)}"""
    label = dict()        
    for bead in bonds:
      for i in range(len(bead)):
        if not bead[i] in label:
          label[bead[i]] = 0 
    counter = 0
    cluster = dict()
    for bond in bonds:
        if label[bond[0]] == label[bond[1]]:
            if label[bond[0]] == 0:
                counter += 1
                label[bond[0]] = counter
                label[bond[1]] = counter
                cluster[counter] = [bond[0], bond[1]]
        else:
            if label[bond[0]] == 0 and label[bond[1]] != 0: 
                label[bond[0]] = label[bond[1]]
                cluster[label[bond[0]]].append(bond[0])
            elif label[bond[1]] == 0 and label[bond[0]] != 0: 
                label[bond[1]] = label[bond[0]] 
                cluster[label[bond[1]]].append(bond[1])
            else:
                minimum = min(label[bond[1]],label[bond[0]])
                maximum = max(label[bond[1]],label[bond[0]])
                cluster[minimum] = cluster[minimum] + cluster[maximum]
                cluster.pop(maximum)
    counter += 1          
    for i in range(number_of_beads):
      if not i in label:
        cluster[counter] = [i]
        counter += 1  
    return cluster                   

def show_beads(*coords, clusters, radius) -> None: 
    """ Crating a plot with bead's clusters in 2D projection
    input parametrs
    *coords - lists coordinates of beads
    clusters - dictionary with clusters {cluster : list(beads)}
    radius - radius of plot circles """
    plt.plot(coords[0], coords[1], 'o', color="orange", markersize=radius)
    for i in clusters.keys():
        for bead in clusters[i]:
                plt.text(coords[0][bead], coords[1][bead], str(i), color="black", fontsize=12,horizontalalignment='center',
                verticalalignment='center')
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    N: Final = 12
    x = np.random.uniform(-4, 4, N)
    y = np.random.uniform(-4, 4, N)
    z = np.random.uniform(-4, 4, N)
    items = list(range(N))
    label = np.zeros(N) 
    clusters = stratification(neighborhood(x, y,  radius = 1.2), N)
    print(clusters)
    show_beads(x, y, clusters=clusters, radius=25 )