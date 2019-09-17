# coding=utf-8
import numpy as np
import random

class SOM:
    def __init__(self, data, shape=None, init='evenly'):
        '''
        shape 竞争层形状
        data  数据集
        init  权值矩阵的初始化方法
        '''
        self.shape=shape
        self.data=np.array(data)
        self.train_data=self.data.copy()

        if shape!=None:
            compete_sum=1 #竞争层总节点数
            for i in shape:
                compete_sum*=i
            self.times=compete_sum*5
        
        if init=='random':
            # 计算边界
            data_t=np.transpose(self.data)
            data_max=data_t[0][0]
            data_min=data_t[0][0]
            for k,v in enumerate(data_t):
                a=max(v)
                b=min(v)
                if a>data_max:
                    data_max=a
                if b<data_min:
                    data_min=b
            # 随机数初始化权值矩阵
            self.weight=np.random.random((compete_sum, self.data.shape[1]))
            self.weight*=data_max-data_min
            self.weight+=data_min
        elif init=='evenly':
            # 计算边界
            data_t=np.transpose(self.data)
            data_max=[data_t[0][0]]*len(data_t)
            data_min=[data_t[0][0]]*len(data_t)
            for k,v in enumerate(data_t):
                a=max(v)
                b=min(v)
                if a>data_max[k]:
                    data_max[k]=a
                if b<data_min[k]:
                    data_min[k]=b
            # 如果未指定竞争层形状
            if shape==None:
                self.shape=[int((data_max[i]-data_min[i])*2) for i in range(len(data_t))]
                print('SOM shape:',self.shape)
                compete_sum=1
                for i in self.shape:
                    compete_sum*=i
                self.times=compete_sum*5
            # 均匀分布初始化权值矩阵
            scale=[np.linspace(data_min[i],data_max[i],self.shape[i]) for i in range(len(data_t))]
            weight=[]
            for i in scale[0]:
                for j in scale[1]:
                    weight.append([i,j])
            self.weight=np.array(weight).astype(float)
        else:
            # 随机取样初始化权值矩阵  
            weight=[]      
            for i in range(compete_sum):
                sample=data[random.randint(0, len(data)-1)]
                weight.append(sample)
            self.weight=np.array(weight).astype(float)

        self.real_weight=self.weight.copy()
        # 控制精度
        # self.weight=np.around(self.weight, decimals=1)

    def run(self):
        # 训练网络
        count=0
        # log=[self.weight.copy()]
        self.stable=False
        while not self.stable and count<len(self.data)*10:
            self.stable=True
            # np.random.shuffle(self.train_data)
            for k,v in enumerate(self.train_data):
                # 调整学习率和领域大小
                self.rate=0.5*np.power(np.e, -2*count/self.times)
                self.radius=int(2*np.power(np.e, -2*count/self.times))

                min_w=np.argmin(self.distance(v))
                for i in self.neighbor(min_w):
                    self.kohonen(i,k)

                count+=1
            # log.append(self.weight.copy())
            print('Training SOM:',count,end='\r')
        if not self.stable:
            print('\n(exceed the upper limit)')
        else:
            print()
        self.real_weight=self.weight
        # 应用网络
        cluster={}
        label=[]
        for k,v in enumerate(self.data):
            min_w=np.argmin(self.distance(v))
            label.append(min_w)
            if min_w in cluster:
                cluster[min_w][1].append(k)
            else:
                cluster[min_w]=[self.weight[min_w], [k]]
        # 模糊化聚类中心
        self.radius=1
        super_cluster_map=dict(zip(cluster.keys(),cluster.keys()))
        for i in cluster.keys():
            for j in self.neighbor(i):
                if j==i:
                    continue
                for k,v in enumerate(self.weight):
                    if (v==self.weight[j]).all() and k in cluster:
                        super_cluster_map[k]=super_cluster_map[i]

        for i in super_cluster_map.keys():
            target=super_cluster_map[i]
            while super_cluster_map[target]!=target:
                target=super_cluster_map[target]
            super_cluster_map[i]=target

            if target!=i:
                cluster[target][1]+=cluster[i][1]
                del cluster[i]
                for j,v in enumerate(label):
                    if v==i:
                        label[j]=target
        return cluster,label
    
    # 计算欧式距离
    def distance(self, vec):
        dis=[]
        for i in self.real_weight:
            dis.append(np.linalg.norm(i-vec))
        return dis

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
        if len(self.shape)==2:
            coordinate=[index/self.shape[1], index%self.shape[1]]
        elif len(self.shape)==3:
            coordinate=[index/(self.shape[1]*self.shape[2]), index%(self.shape[1]*self.shape[2])/self.shape[2],index%self.shape[2]]
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
        new=self.weight[w]+self.rate*(self.train_data[d]-self.weight[w])
        if np.linalg.norm(new-self.weight[w])>0.142:
            self.stable=False
        self.weight[w]=2*np.around(new*0.5, decimals=1)#限制分辨率
        self.real_weight[w]=self.real_weight[w]+self.rate*(self.train_data[d]-self.real_weight[w])