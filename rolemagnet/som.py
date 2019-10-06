# coding=utf-8
import numpy as np
import random

# 参数
Resolution=0.2
Density=2
Stable_line=2*Resolution*Resolution
Learn_rate=0.5
Max_size=10

class SOM:
    def __init__(self, data, shape=None, init='evenly'):
        '''
        shape 竞争层形状
        data  数据集
        init  权值矩阵的初始化方法
        '''
        self.shape=shape
        
        # 计算边界
        data_t=np.transpose(data)
        data_max=[data_t[0][0]]*len(data_t)
        data_min=[data_t[0][0]]*len(data_t)
        for k,v in enumerate(data_t):
            a=max(v)
            b=min(v)
            if a>data_max[k]:
                data_max[k]=a
            if b<data_min[k]:
                data_min[k]=b
        # 压缩数据
        self.data=np.array(data)
        size=[data_max[i]-data_min[i] for i in range(len(data_t))]
        data_max=np.array(data_max)
        data_min=np.array(data_min)
        size=np.array(size)
        if max(size)>Max_size:
            coeff=Max_size/max(size)
            self.data*=coeff
            size*=coeff
            data_max*=coeff
            data_min*=coeff

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
            # 如果未指定竞争层形状
            if shape==None:
                self.shape=[int(i*2) for i in size]
                print('SOM shape:',self.shape)
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

        # 估计训练次数
        compete_sum=1 #竞争层总节点数
        for i in self.shape:
            compete_sum*=i
        self.times=compete_sum*5

        # 初始化获胜神经元
        self.winner=[0]*len(data)
        for k,v in enumerate(self.data):
            self.update_winner(k)

    def run(self):
        # 训练网络
        count=0
        train_round=0
        # log=[self.weight.copy()]
        self.stable=False
        self.line=Stable_line
        while not self.stable:
            self.stable=True
            # np.random.shuffle(self.train_data)
            for k,v in enumerate(self.data):
                # 调整学习率和邻域大小
                self.rate=0.5*np.power(np.e, -3*count/self.times)
                self.radius=int(2*np.power(np.e, -2*count/self.times))

                self.update_winner(k)
                winner=self.winner[k]
                for i in self.neighbor(winner):
                    self.kohonen(i,k)

                count+=1
            # log.append(self.weight.copy())
            print('Training SOM:',count,end='\r')
            train_round+=1
            self.line+=np.power(np.e, train_round-10)# 调整稳定条件
        print()

        # 应用网络
        cluster={}
        label=[]
        for k,v in enumerate(self.data):
            self.update_winner(k)
            winner=self.winner[k]
            label.append(winner)
            if winner in cluster:
                cluster[winner][1].append(k)
            else:
                cluster[winner]=[self.weight[winner], [k]]
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
    
    # 计算获胜神经元
    def update_winner(self, index):
        vec=self.data[index]
        delta=vec-self.weight[self.winner[index]]
        min_dis=delta.dot(delta)
        winner=[]
        for i,v in enumerate(self.weight):
            delta=v-vec
            distance=delta.dot(delta)
            if distance<min_dis:
                min_dis=distance
                winner=[i]
            elif distance==min_dis:
                winner.append(i)
        if self.winner[index] not in winner:
            self.winner[index]=winner[0]

    # 获取邻域内节点的序号
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
                    if abs(i-coo[0])+abs(j-coo[1])<=self.radius:#半径小于2，切比雪夫距离可以代替欧式距离
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
                        if abs(i-coo[0])+abs(j-coo[1])+abs(k-coo[2])<=self.radius:
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
        new=self.weight[w]+self.rate*(self.data[d]-self.weight[w])
        delta=new-self.weight[w]
        if delta.dot(delta)>self.line:
            self.stable=False
        self.weight[w]=2*np.around(new*0.5, decimals=1)#限制分辨率