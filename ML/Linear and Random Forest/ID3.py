import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

data = pd.read_csv('.\ID3.csv')
features = [feat for feat in data]
features.remove("Answer")

class Node:
    def __init__(self):
        self.children = []
        self.values = ""
        self.isLeaf = False
        self.pred = ""

def entropy(examples):
        pos = 0.0
        neg = 0.0
        for _, row in examples.iterrows():
            if row["Answer"] == "yes":
                pos += 1
            else:
                neg += 1
        if pos == 0.0 or neg == 0.0:
            return 0.0
        else:
            p = pos / (pos+neg)
            n = neg / (pos+neg)
            return -(p*math.log(p,2)+n*math.log(n,2))
def info_gain(examples, attr):
        uniq = np.unique(examples[attr])
        gain = entropy(examples)
        for u in uniq:
            subdata = examples[examples[attr]==u]
            sub_e = entropy(subdata)
            gain -= (float(len(subdata)) / float(len(examples))) * sub_e    
        return gain
def ID3(examples, attrs):
        root = Node()

        max_gain = 0
        max_feat = ""
        for features in attrs:
            gain = info_gain(examples,features)
            if gain>max_gain:
                max_gain=gain
                max_feat=features
        root.values = max_feat
        uniq = np.unique(examples[max_feat])
        for u in uniq:
            subdata = examples[examples[max_feat]==u]
            if entropy(subdata) == 0.0:
                newNode = Node()
                newNode.isLeaf = True
                newNode.values = u
                newNode.pred = np.unique(subdata["Answer"])
                root.children.append(newNode)
            else:
                dummyNode = Node()
                dummyNode.values = u
                new_attrs = attrs.copy()
                new_attrs.remove(max_feat)
                child = ID3(subdata,new_attrs)
                dummyNode.children.append(child)
                root.children.append(dummyNode)
        return root
def printTree(root: Node, depth=0):
    for i in range(depth):
        print("\t", end="")
    print(root.values, end="")
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    for child in root.children:
        printTree(child, depth+1)
root = ID3(data,features)
printTree(root)