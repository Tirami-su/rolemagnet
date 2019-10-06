# coding=utf-8
import numpy as np
import networkx as nx
import queue
from .graphwave import *

import multiprocessing, time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .som import SOM

def sub_graph(G, node, g, que, checked):
    que.put(node)
    while not que.empty():
        search_adj(G, que.get(), g, que, checked)
        
    checked.clear()
        
def search_adj(G, node, g, que, checked):
    for k,v in G[node].items():
        if k in checked:
            continue
        g.add_weighted_edges_from([(node, k, v['weight'])])
        que.put(k)    
        
    checked[node]=1#标记该点搜索完毕

def embed(G, rev_G, sample, index, curNode):
    search_queue=queue.Queue()  
    checked={}
    
    # 聚图
    gat = nx.Graph()
    gat.add_nodes_from(G)      
    sub_graph(rev_G, curNode, gat, search_queue, checked)
    chi_gat,_,_ = graphwave_alg(gat, sample, node=index, verbose=True)
    
    # 散图
    dif = nx.Graph()
    dif.add_nodes_from(G)
    sub_graph(G, curNode, dif, search_queue, checked)
    chi_dif,_,_ = graphwave_alg(dif, sample,  node=index, verbose=True)

    # 把计算结果放入队列
    chi_queue.put([index, np.concatenate((chi_gat[index],chi_dif[index]), axis=0)])

chi_queue = multiprocessing.Queue()# 子进程的输出队列

def role_magnet(G, balance=None, sample=np.linspace(0,100,25), shape=None):
    '''
    参数
    G       图，networkx的Graph/DiGraph
    balance 出入流量差
    sample  采样点
    shape   SOM竞争层的形状
    -----------------------------------
    返回值
    vec     节点的向量表示
    role    聚类结果，key:角色代号，value:[聚类中心位置，属于该角色的成员]
    label   每个点对应的角色代号
    '''
    rev_G = G.reverse()

    # 创建子进程，每个进程计算一个点
    for index,curNode in enumerate(G.nodes):
        proc = multiprocessing.Process(target=embed, args=(G,rev_G,sample,index,curNode))
        proc.start()

    finished=0
    total=len(G.nodes)
    chi=np.empty((total, len(sample)*8))
    count=0
    character=['/','-','\\','-']

    while finished != total:
        # 主进程从队列接收各点的嵌入结果
        while not chi_queue.empty():
            res=chi_queue.get()
            chi[res[0]]=res[1]
            finished+=1

        # 输出进度
        print('Embedding: %5.2f%%  %c' %(finished/total*100, character[count%4]), end='\r')
        count+=1
        time.sleep(1)
    print()
    
    # 降到二维，加上流量差
    reduced=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(chi))
    vec=reduced
    if not balance is None:
        # 调整流量差的分布
        balance=np.array(balance).reshape(len(balance),1)
        chi_t=np.transpose(reduced)
        balance=(max(chi_t[0])-min(chi_t[0])+max(chi_t[1])-min(chi_t[1]))/2*balance
        vec=np.concatenate((reduced, balance), axis=1)
        # 再降到二维
        reduced=PCA(n_components=2).fit_transform(StandardScaler().fit_transform(vec))

    # mid_time=time.perf_counter()
    som=SOM(reduced, shape)
    role,label=som.run()
    return vec,role,label