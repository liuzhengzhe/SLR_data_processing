'''
Created on 2014-9-21

@author: lenovo
'''
import matplotlib.pyplot as plt
def plot_curve(index,value):
    plt.plot(value)
    plt.show()
    
def plot_a_graph(labels,data):

    hog_length=[];
    with open('test.txt','r') as f:
        for line in f:
            hog_length.append(map(float,line.split(' ')))

    f.close()
    index=20;
    ind=labels.index(index)
    left=[];
    height=[];

    front=0;
    while(labels[front]!=labels[ind]):
        front=front+1
    

    while(labels[front]==labels[ind]):
        plt.figure(figsize=(5,5));
        for i in range(192,207):
            left.append(i);
            height.append(data[ind][i]*hog_length[front][0]);
        for i in range(207,222):
            left.append(i);
            height.append(data[ind][i]*hog_length[front][1]);
        plt.bar(left, height)  
        plt.ylabel(front)
        front=front+1;
    plt.show();