import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
import networkx as nx
from igraph import *
from tqdm import tqdm
import random

g = nx.read_edgelist('data/Train_1000_2', nodetype=int)


# def IC(g, S, p=0.2, mc=1000):
#     spread = []
#     for i in range(mc):
#
#         # Simulate propagation process
#         new_active, A = S[:], S[:]
#         while new_active:
#
#             # For each newly active node, find its neighbors that become activated
#             new_ones = []
#             for node in new_active:
#                 # Determine neighbors that become infected
#                 # np.random.seed(i)
#                 success = np.random.uniform(0, 1, len(list(g.neighbors(node)))) < p
#                 new_ones += list(np.extract(success, list(g.neighbors(node))))
#
#             new_active = list(set(new_ones) - set(A))
#
#             # Add newly activated nodes to the set of activated nodes
#             A += new_active
#
#         spread.append(len(A))
#
#     return np.mean(spread)

def IC(G, initial_node, p=0.2, mc=1000):
    """ 独立级联传播模型 """
    spread = []
    for i in range(mc):
        activated = set(initial_node)  # 初始激活的节点集合
        to_activate = set(initial_node)  # 要激活的节点

        while to_activate:  # 当还有要激活的节点时
            new_to_activate = set()
            for node in to_activate:
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in activated:  # 只处理未激活的节点
                        if random.random() < p:  # 以概率决定激活
                            new_to_activate.add(neighbor)
                            activated.add(neighbor)

            to_activate = new_to_activate
        spread.append(len(activated))
    return np.mean(spread)


def celf(g, k, p=0.1, mc=1000):
    start_time = time.time()
    marg_gain = [IC(g, [node], p, mc) for node in tqdm(range(g.number_of_nodes()))]

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(range(g.number_of_nodes()), marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [g.number_of_nodes()], [time.time() - start_time]

    for _ in tqdm(range(k - 1)):

        check, node_lookup = False, 0

        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1

            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IC(g, S + [current], p, mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)


start_time = time.time()

# Run algorithms
celf_output = celf(g, 15, p=0.2, mc=1000)
end_time = time.time()
runningtime1 = end_time - start_time
print("总时间：", runningtime1)
# Print results
print("celf output:   " + str(celf_output[0]))
print("15-set spread:  " + str(celf_output[1]))
with open('15_set.txt', 'w') as f:
    f.write('selected_nodes: ')
    f.write(str(celf_output[0])+'\n')
    f.write('spread: ' + str(celf_output[1]))
