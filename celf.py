import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
import networkx as nx
from igraph import *
from tqdm import tqdm

g = nx.read_edgelist('data/Train_1000_2', nodetype=int)


def IC(g, S, p=0.2, mc=1000):
    spread = []
    for i in range(mc):

        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0, 1, len(list(g.neighbors(node)))) < p
                new_ones += list(np.extract(success, list(g.neighbors(node))))

            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        spread.append(len(A))

    return np.mean(spread)

ifs = IC(g,[6])

def celf(g, k, p=0.1, mc=1000):
    start_time = time.time()
    marg_gain = [IC(g, [node], p, mc) for node in range(g.number_of_nodes())]

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
celf_output = celf(g, 50, p=0.2, mc=1000)
end_time = time.time()
runningtime1 = end_time - start_time
print("总时间：", runningtime1)
# Print results
print("celf output:   " + str(celf_output[0]))
