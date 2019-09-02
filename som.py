import numpy as np
import random

class SOM:
    def __init__(self, shape, data, init='random'):
        '''
        shape 竞争层形状
        data  数据集
        init  权值矩阵的初始化方法
        '''
        self.shape=shape
        self.data=np.array(data)
        
        compete_sum=1 #竞争层总节点数
        for i in shape:
            compete_sum*=i
            
        self.times=compete_sum*10
        
        if init=='random':
            # 随机数初始化权值矩阵
            self.weight=np.random.random((compete_sum, self.data.shape[1]))
        else:
            # 随机取样初始化权值矩阵  
            weight=[]      
            for i in range(compete_sum):
                sample=data[random.randint(0, len(data)-1)]
                weight.append(sample)
            self.weight=np.array(weight).astype(float)
        
    def run(self):
        # 训练
        count=0
        while count<self.times:
            for k,v in enumerate(self.data):
                # 调整学习率和领域大小
                self.rate=0.8*np.power(np.e, -2*count/self.times)
                self.radius=int(self.shape[0]*np.power(np.e, -2*count/self.times))
                count+=1

                max_w=self.weight.dot(v).argmax()
                nb=self.neighbor(max_w)
                for i in nb:
                    self.kohonen(i,k)
        # 测试
        cluster={}
        label=[]
        for k,v in enumerate(self.data):
            max_w=self.weight.dot(v).argmax()
            label.append(max_w)
            if max_w in cluster:
                cluster[max_w][1].append(k)
            else:
                cluster[max_w]=[self.weight[max_w], [k]]
        return cluster,label
    
    # 获取领域内节点的序号
    def neighbor(self, node):
        nb=[]
        if len(self.shape)==1:
            left=node-self.radius if node>self.radius else 0
            right=node+self.radius if node+self.radius<self.shape[0] else self.shape[0]-1
            nb=range(left, right+1)
        elif len(self.shape)==2:
            coo=self.index2coordinate(node)
            up=coo[0]-self.radius if coo[0]>self.radius else 0
            down=coo[0]+self.radius if coo[0]+self.radius<self.shape[0] else self.shape[0]-1
            left=coo[1]-self.radius if coo[1]>self.radius else 0
            right=coo[1]+self.radius if coo[1]+self.radius<self.shape[1] else self.shape[1]-1
            for i in range(up, down+1):
                for j in range(left, right+1):
                    if np.linalg.norm(np.array(coo)-np.array((i,j)))<=self.radius:
                        nb.append(self.coordinate2index((i,j)))
        elif len(self.shape)==3:
            coo=self.index2coordinate(node)
            up=coo[0]-self.radius if coo[0]>self.radius else 0
            down=coo[0]+self.radius if coo[0]+self.radius<self.shape[0] else self.shape[0]-1
            left=coo[1]-self.radius if coo[1]>self.radius else 0
            right=coo[1]+self.radius if coo[1]+self.radius<self.shape[1] else self.shape[1]-1
            front=coo[2]-self.radius if coo[2]>self.radius else 0
            back=coo[2]+self.radius if coo[2]+self.radius<self.shape[2] else self.shape[2]-1
            for i in range(up, down+1):
                for j in range(left, right+1):
                    for k in range(front, back+1):
                        if np.linalg.norm(np.array(coo)-np.array((i,j,k)))<=self.radius:
                            nb.append(self.coordinate2index((i,j,k)))
        return nb

    # 序号转坐标
    def index2coordinate(self, index):
        coordinate=[index/self.shape[0], index%self.shape[0]/self.shape[1]]
        if len(self.shape)==3:
            coordinate.append(index%self.shape[0]%self.shape[1]/self.shape[2])
        return [int(i) for i in coordinate]

    # 坐标转序号
    def coordinate2index(self, coordinate):
        index=0
        weight=1
        for i in reversed(range(len(self.shape))):
            index+=coordinate[i]*weight
            weight*=self.shape[i]
        return index

    # Kohonen学习规则
    def kohonen(self, w, d):
        self.weight[w]+=self.rate*(self.data[d]-self.weight[w])