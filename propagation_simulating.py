import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


G = nx.read_edgelist("data/Train_1000_2", nodetype=int)


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


marg_gain = [IC(G, [node]) for node in tqdm(range(G.number_of_nodes()))]

# Create the sorted list of nodes and their marginal gain
Q = sorted(zip(range(G.number_of_nodes()), marg_gain), key=lambda x: x[1], reverse=True)
with open('nodes_if.txt', 'w') as f:
    for x, y in Q:
        f.write(f"{x} {y}\n")
print(Q)


# # 传播模拟
# activated_nodes = independent_cascade(G, initial_node, uniform_probability)
#
# # 打印结果
# print(f"初始激活节点: {initial_node}")
# print(f"激活的节点: {activated_nodes}")
#
# # 可视化网络
# pos = nx.spring_layout(G)
# plt.figure(figsize=(16, 12))
#
# # 绘制网络
# nx.draw(G, pos, with_labels=False, edge_color='lightgray', node_color='lightgreen', node_size=10, font_size=10)
#
# # 高亮激活节点
# nx.draw_networkx_nodes(G, pos, nodelist=activated_nodes, node_color='red',node_size=10)
# plt.title("Independent Cascade Model (Activated Nodes in Red)")
# plt.show()