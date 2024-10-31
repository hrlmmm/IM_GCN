import networkx as nx
import matplotlib.pyplot as plt

# 训练BA网络生成
n = [1000]
k = [2]
for i in n:
    for j in k:
        BAgraph = nx.random_graphs.barabasi_albert_graph(i, j)  # 节点数，平均度数
        nx.write_edgelist(BAgraph, '/data/Train_'+str(i)+'_'+str(j), data=False)

测试BA网络生成
n = [10000]
k = [2, 4, 8]
for i in n:
    for j in k:
        BAgraph = nx.random_graphs.barabasi_albert_graph(i, j)  # 节点数，平均度数
        nx.write_edgelist(BAgraph, 'Test_'+str(i)+'_'+str(j), data=False)
