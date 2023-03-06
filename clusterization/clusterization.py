import numpy as np
import matplotlib.pyplot as plt
from typing import (Final, List, Dict, Tuple)


def distance(*args: float) -> float:
    """ Decart distance """
    return np.sqrt(np.sum(np.square(args)))


def neighbourhood(*coords: np.ndarray, radius: float) -> List[int]:
    """ 
    Return list of neighboring beads 

    input:
    radius - minimal distance between neighboring beads
    *coords - coordinates of beads

    output:
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
                bonds.append((i, j))
    return bonds


def stratification(num_beads: int, bonds: List[Tuple[int]]) -> Dict:
    """ 
    Define claster's id 
    input:
    num_beads - number of all beads
    bonds - list of beads within a radius from each other 

    output:
    Dict{cluster : list(bead_1, .., bead_n)}
    """
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
                minimum = min(label[bond[1]], label[bond[0]])
                maximum = max(label[bond[1]], label[bond[0]])
                cluster[minimum] = cluster[minimum] + cluster[maximum]
                for clust in cluster[maximum]:
                    label[clust] = minimum
                cluster.pop(maximum)
                counter -= 1
    counter = 0
    for clust in cluster:
        counter += 1
        clust = counter
    for i in range(num_beads):
        if not (i in label):
            counter += 1
            cluster[counter] = [i]
    return cluster


def show_beads(*coords: np.ndarray, clusters: Dict, radius: float) -> None:
    """ 
    Crating a plot with bead's clusters in 2D projection
    input:
    *coords - lists of the coordinates of beads
    clusters - {cluster : list(bead_1, .., bead_n)}
    radius - radius of plot circles 
    """
    color = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    for clust, col in zip(clusters, color):
        plt.plot(coords[0][clusters[clust]], coords[1]
                 [clusters[clust]], 'o', color=col, markersize=radius)
    for i in clusters.keys():
        for bead in clusters[i]:
            plt.text(coords[0][bead], coords[1][bead], str(i), color="black", fontsize=12, horizontalalignment='center',
                     verticalalignment='center')
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    N: Final = 37
    rnd = 4
    x = np.random.uniform(-rnd, rnd, N)
    y = np.random.uniform(-rnd, rnd, N)
    z = np.random.uniform(-rnd, rnd, N)
    clusters = stratification(N, neighbourhood(x, y, z, radius=1.2))
    print(clusters)
    show_beads(x, y, clusters=clusters, radius=25)
