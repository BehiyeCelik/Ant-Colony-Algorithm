# %%
import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.random as rnd
import tsplib95
import tsplib95.distances as distances
from alns import ALNS, State
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime
import sys
import numpy
import random as rn
import numpy as np
from numpy.random import choice as np_choice


# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
SEED = 7654

# %%
# Graph named 'xqf131' with 131 nodes and 8646 edges
DATA = tsplib95.load('xqf131.tsp')
CITIES = list(DATA.node_coords.keys())

# %%
# Precompute the distance matrix - this saves a bunch of time evaluating moves.
# + 1 since the cities start from one (not zero).
COORDS = DATA.node_coords.values()
DIST = np.empty((len(COORDS) + 1, len(COORDS) + 1))
print(DIST.shape)
for row, coord1 in enumerate(COORDS, 1):
    for col, coord2 in enumerate(COORDS, 1):
        DIST[row, col] = distances.euclidean(coord1, coord2)

x_DIST = np.empty((len(COORDS), len(COORDS)))
for row, coord1 in enumerate(COORDS, 0):
    for col, coord2 in enumerate(COORDS, 0):
        x_DIST[row, col] = distances.euclidean(coord1, coord2)
        
SOLUTION = tsplib95.load('xqf131.opt.tour')
OPTIMAL = DATA.trace_tours(SOLUTION.tours)[0]

print(f"Total optimal tour length is {OPTIMAL}.")

# %%
numpy.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

# %%
x_DIST.shape
print(x_DIST)

# %%
x_DIST = x_DIST.round(3)

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

x_DIST = trunc(x_DIST, 1)

for i in range(131):
    for j in range(131):
        if x_DIST[i][j] == 0:
            x_DIST[i][j] = np.inf

print(x_DIST)


# %%
numpy.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

# %%
for i in range(131):
    for j in range(131):
        if x_DIST[i][j] == 0:
            x_DIST[i][j] = np.inf

# %%
print(COORDS)

# %%
def x_draw_graph(graph, only_nodes=False, coords_value = DATA.node_coords, label = True):
    """
    Helper method for drawing TSP (tour) graphs.
    """
    print("nodes\n")
    graph_nodes = graph.nodes
    print(graph_nodes)
    list_graph_nodes = list(graph_nodes)
    print(len(list_graph_nodes))

    print("edges\n")
    graph_edges = graph.edges
    print(graph_edges)
    list_graph_edges = list(graph_edges)
    print(len(list_graph_edges))
    
    print("coords_value\n")
    print(len(coords_value))
    print(coords_value)
    
    
    fig, ax = plt.subplots(figsize=(12, 6))

    if only_nodes:
        nx.draw_networkx_nodes(graph, coords_value, node_size=150, ax=ax)
        if label:
            nx.draw_networkx_labels(graph, coords_value, labels={i: str(i) for i in range(0, len(graph.nodes) + 1)}, font_size=8, font_color='white')
    else:
        nx.draw_networkx(graph, coords_value, node_size=150, with_labels=False, ax=ax)
        if label:
            nx.draw_networkx_labels(graph, coords_value, labels={i: str(i) for i in range(0, len(graph.nodes) + 1)}, font_size=8, font_color='white')


# %%
print(DATA.get_graph())

# %%
n_ants=0
n_best=0
n_iterations=0
decay=0
alpha=0
beta=0
pheromone=0
all_inds=0

# %%
def pick_move(pheromone, dist, visited):
   pheromone = np.copy(pheromone)
   #Make zero if the path has been visited
   pheromone[list(visited)] = 0

   #Ant makes a decision on what city to go using this formula
   row = pheromone ** alpha * (( 1.0 / dist) ** beta)

   #Probability formula
   norm_row = row / row.sum()

   #Move randomly using probability (select path to go using probability)
   #p=probability
   #Get index of an element that has bigger probability 
   move = np_choice(all_inds, 1, p=norm_row)[0]
   #print(move)

   #Return path that randomly selected
   return move

# %%
def spread_pheronome(all_paths, n_best, shortest_path): # Q/Lk - Quantity of pheromone:
   global pheromone, distances
   # sorted a path form small to big (with values to be shorted are values on second collumn)
   sorted_paths = sorted(all_paths, key=lambda x: x[1])

   for path, dist in sorted_paths[:n_best]:
      #print("spread pheronome: {0} ## {1}".format(path, dist))
      for move in path:
         #print (move)
         # ant deposits a pheromone on the way that its travelled 
         # the amount of pheromone that the ant deposit is (1/distances between 2 cities)
         pheromone[move] += 1.0 / distances[move]

# %%
def gen_path_dist(path):
   global distances
   total_dist = 0
   for ele in path:
      total_dist += distances[ele]
   return total_dist

# %%
def gen_path(start):
   global pheromone, distances
   path = []
   #Start path to 0
   visited = set()
   visited.add(start)
   #prev = start
   prev = start
   for i in range(len(distances) - 1):
      move = pick_move(pheromone[prev], distances[prev], visited)
      #Append path
      path.append((prev, move))
      #Change previous path to move path after append path
      prev = move
      #add Path that has been moved to visited, so the path that has
      #been visited can be made to zero
      visited.add(move)
   path.append((prev, start)) # going back to where we started    
   return path

# %%
def gen_all_paths():
   all_paths = []
   for i in range(n_ants):
      path = gen_path(0) #0 is start
      all_paths.append((path, gen_path_dist(path)))
   return all_paths

# %%
def aco(distances_, n_ants_, n_best_, n_iterations_, decay_, alpha_=1, beta_=1, threshold = 0):
   print("Starting\n")
   global distances, n_ants, n_best, n_iterations, decay, alpha, beta, pheromone, all_inds
   distances = distances_
   n_ants = n_ants_
   n_best = n_best_
   n_iterations = n_iterations_
   decay = decay_
   alpha = alpha_
   beta = beta_
   pheromone = np.ones(distances_.shape) / 10
   all_inds=range(len(distances_))

   shortest_path = None
   all_time_shortest_path = ("placeholder", np.inf)
   for i in range(n_iterations_):
      #Get all paths
      all_paths = gen_all_paths()
      
      #Spread Pheromone
      spread_pheronome(all_paths, n_best_, shortest_path=shortest_path)
      
      #Get the shortest path in all paths, based on its distance (x[1])
      shortest_path = min(all_paths, key=lambda x: x[1])
    
      #print("shortest path : ## {0}".format(shortest_path))
      
      # if total distance < infinity then :
      # all_time_shortest_path = shortest_path
      if shortest_path[1] < all_time_shortest_path[1]:
        all_time_shortest_path = shortest_path
        if(shortest_path[1] < threshold):
            print(f'all_time_shortest_path {all_time_shortest_path}')
            return all_time_shortest_path, shortest_path
      # pheromone decay (pheromone * decay rate)           
      pheromone = pheromone * decay            

      #return all_time_shortest_path #the shortest path that founded by the ants
    
    #print(f'all_time_shortest_path' + all_time_shortest_path)
   print(f'all_time_shortest_path {all_time_shortest_path}')
   return all_time_shortest_path, shortest_path

# %%
#instance distances
distances = np.array([[np.inf, 2, 2, 5, 7],
                      [2, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 3],
                      [0, 8, 1, np.inf, 2],
                      [7, 2, 3, 2, np.inf]])

# %%
class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)          
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone = self.pheromone * self.decay            
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # going back to where we started    
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

# %%
ant_colony = AntColony(x_DIST, 100, 100, 50, 0.95, 1, 1) 
shortest_path = ant_colony.run()

# %%
shortest_path

# %%
# dist_matrix
x_DIST = np.nan_to_num(x_DIST)

# %%
x_DIST.shape

# %%
for i in range(131):
    for j in range(131):
        if x_DIST[i][j] < 0:
            print(x_DIST[i][j])

# %%
all_time_shortest_path, shortest_path = aco(x_DIST, 100, 100, 50, 0.1, 1, 1, 0)

# %%
test = all_time_shortest_path
ar = np.array(shortest_path[0])
ar_list = np.ndarray.tolist(ar)
ar_list = list(ar_list)
ar_list.sort()
x_nodes = list(range(0,130))
x_edges = ar_list[1:131]
print(len(ar_list))
x_edges_dict = dict(x_edges)
x_obj = sum(x_DIST[node, x_edges_dict[node]] for node in x_nodes[1:len(x_nodes) - 1])
print(x_obj)
x_list_COORDS_dict = list(range(0,132))
x_list_COORDS = list(COORDS)
x_list_COORDS.append([0,0])
x_dict = dict(zip(x_list_COORDS_dict,x_list_COORDS))
XG = nx.Graph()
XG.add_nodes_from(x_nodes[0:131])
XG.add_edges_from(ar_list[0:131])
x_draw_graph(XG, only_nodes=False, label =True, coords_value = x_dict)


