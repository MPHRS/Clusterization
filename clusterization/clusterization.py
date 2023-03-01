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

def stratification(bonds: List) -> Dict: 
    """ Define claster's id """
    label = {}
    dct = set()
    [dct.add(bead[0]) for bead in bonds]
    [dct.add(bead[1]) for bead in bonds]
    for num in dct:
        label[num] = 0
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
                cluster[label[bond[0]]] = bond[0]
            elif label[bond[1]] == 0 and label[bond[0]] != 0: 
                label[bond[1]] = label[bond[0]] 
                cluster[label[bond[1]]] = bond[1]
            else:
                minimum = min(label[bond[1]],label[bond[0]])
                maximum = max(label[bond[1]],label[bond[0]])
                cluster[minimum] = cluster[minimum] + cluster[maximum]
                cluster.pop(maximum)
                
                  
        
    return cluster              
    
if __name__ == '__main__':
    np.random.seed(42)
    N: Final = 12
    x = np.random.uniform(-4, 4, N)
    y = np.random.uniform(-4, 4, N)
    items = list(range(N))
    label = np.ndarray([0] * N)
    clusters = stratification(neighborhood(x, y, radius = 1.2))
    print(clusters)
    # print(neighborhood(x, y, radius=1.2))
    # plt.plot(x, y, 'o', color="orange", markersize=12)
    # for i in range(len(x)):
    #     plt.text(x[i], y[i], str(i), color="black", fontsize=12)
    # plt.show()