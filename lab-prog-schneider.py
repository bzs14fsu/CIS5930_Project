#Installs and Imports

#These depend on your environment. May be unnecessary. I needed them on Colab.
#!pip install dgl
#%load CSmodel.py
#from CSmodel import CorrectAndSmooth

#designing neural networks
import dgl
import torch as tch
import keras as ks
import scipy as spy
import networkx as nx

#data manipulation
import csv
import math
import numpy as np

#graphing results
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data

#for mapping data to range [0, 1]
def pos(conv):
  return (conv + 1) * 0.5

#training data
train_truth = []  #contains ground truth classification
train_inter = []   #intermediate lists
train_string = []
train_floats = []

with open('zip_train.txt', newline = '\n') as csvfile:
  terms = csv.reader(csvfile, delimiter = ' ')
  for term in terms:
    term.remove('')
    train_truth.append(int(term[0]))
    train_string.append(term[1:])

for row in train_string:
  train_floats = []
  for val in row:
    train_floats.append(pos(float(val)))
  train_inter.append(np.reshape(train_floats, (16, 16)))

print("Training classifications: ", train_truth)

#test data
test_truth = []   #contains ground truth classification
test_inter = []    #intermediate lists
test_string = []
test_floats = []

with open('zip_test.txt', newline = '\n') as csvfile:
  terms = csv.reader(csvfile, delimiter = ' ')
  for term in terms:
    term.remove('')
    test_truth.append(int(float(term[0])))
    test_string.append(term[1:])

for row in test_string:
  test_floats = []
  for val in row:
    test_floats.append(pos(float(val)))
  test_inter.append(np.reshape(test_floats, (16, 16)))

print("Testing classifications: ", test_truth)

#full dataset, as numpy arrays
full_truth = train_truth + test_truth

train_vals = np.array(train_inter)
test_vals = np.array(test_inter)
full_vals = np.array(train_inter + test_inter)

print("All classifications: ", full_truth)

#testing results
print(len(train_truth), " training samples")
print(len(test_truth), " test samples")
print(len(full_truth), " total samples")

##Creating Graph(s)
#distances = []
#edgelist_in = []
#edgelist_out = []
#idx, idx2 = 0, 0
#sums = 0
#
#for row in full_vals:
#  #distance from a node to all others
#  for comprow in full_vals:
#    sums = np.sqrt(np.sum(np.square(row - comprow)))
#    distances.append([sums, idx2])
#    idx2 += 1
#  
#  #take shortest 7 distances
#  distances.pop(idx)
#  distances.sort()
#  edgelist_out = edgelist_out + ([idx] * 7)
#  for x in distances[0:7]:
#    edgelist_in.append(x[1])
#
#  #reset counters
#  idx += 1
#  idx2 = 0
#  sums = 0
#  distances = []
#
#save edge list
#np.savetxt('edgelist.txt', [edgelist_out, edgelist_in])

#Loading Edge List
edgelist = np.array([])

edgelist = np.loadtxt('edgelist.txt')
edgelist_out = edgelist[0].astype(int)
edgelist_in = edgelist[1].astype(int)

#Basic Graphing

#basic graph
begin, end = tch.tensor(edgelist_out), tch.tensor(edgelist_in)
dig_graph = dgl.graph((begin, end))
dig_graph = dgl.to_bidirected(dig_graph)

#adding features to nodes
dig_graph.ndata['pixels'] = tch.tensor(np.reshape(full_vals, (9298, 256)), dtype = tch.float32)

#Super Graph

super_graph = np.zeros((10, 10))
edge_in, edge_out = dig_graph.edges()

for i in range(len(edge_out)):
  super_graph[full_truth[edge_in[i]]][full_truth[edge_out[i]]] += 1

#remove duplicate edges
for i in range(10):
  super_graph[i][i] /= 2

#printing super graph
super_graph_out = pd.DataFrame(super_graph, range(10), range(10))
sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
sn.heatmap(super_graph_out, annot = True, fmt = 'g')
plt.xlabel('Node')
plt.ylabel('Node')
plt.title('Super Graph')
plt.show()

#Task 1 - Convolutional NN

ConvNN = ks.models.Sequential()
ConvNN.add(ks.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (16, 16, 1)))
ConvNN.add(ks.layers.MaxPooling2D((2, 2)))
ConvNN.add(ks.layers.Conv2D(64, (3, 3), activation = 'relu'))

ConvNN.add(ks.layers.Flatten())
ConvNN.add(ks.layers.Dense(100, activation='relu'))
ConvNN.add(ks.layers.Dense(10, activation='softmax'))

ConvNN.compile(optimizer = 'adam', loss = ks.losses.SparseCategoricalCrossentropy(), metrics = ['acc'])

history = ConvNN.fit(train_vals.tolist(), train_truth, epochs = 40, validation_data = (test_vals.tolist(), test_truth))

plt.plot(history.history['acc'], label = 'train acc')
plt.plot(history.history['val_acc'], label = 'test acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.show()

Conv_conf = ConvNN.predict(test_vals).argmax(1)

#Confusion Matrix
Conv_mat = np.zeros((10, 10))
for i in range(len(Conv_conf)):
  Conv_mat[Conv_conf[i]][test_truth[i]] += 1

Conv_mat_out = pd.DataFrame(Conv_mat, range(10), range(10))
sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
sn.heatmap(Conv_mat_out, annot = True, fmt = 'g')
plt.xlabel('Ground Truth')
plt.ylabel('Deep Neural Network (test)')
plt.title('Confusion - Ground Truth vs. DNN')
plt.show()


Conv_pred = ConvNN.predict(full_vals).argmax(1)

##Task 2 - Correct and Smooth
#CS_graph = dig_graph
#
##THIS METHOD IS NOT MY CODE
#cs = CorrectAndSmooth(num_correction_layers = 20, correction_alpha = 0.5,
#                          correction_adj = 'DAD', num_smoothing_layers = 20,
#                          smoothing_alpha = 0.5, smoothing_adj = 'DAD')
#
##create mask and run methods
#mask = []
#mask = [True] * 2007 + [False] * 7291
#y_soft = cs.correct(CS_graph, ConvNN, train_truth, mask)
#y_soft = cs.smooth(CS_graph, y_soft, train_truth, mask)

#Task 3 - Spectral-Based GNN

spec_dig_graph = dig_graph

#adapted from tutorial on DGL website
class GCN(tch.nn.Module):
  def __init__(self, in_f, h_f1, h_f2, classes):
    super(GCN, self).__init__()
    self.layer1 = dgl.nn.GraphConv(in_f, h_f1)
    self.layer2 = dgl.nn.GraphConv(h_f1, h_f2)
    self.layer3 = dgl.nn.GraphConv(h_f2, classes)

  def forward(self, spec_dig_graph, in_feat):
    result = self.layer1(spec_dig_graph, in_feat)
    result = tch.nn.functional.relu(result)
    result = self.layer2(spec_dig_graph, result)
    result = tch.nn.functional.relu(result)
    result = self.layer3(spec_dig_graph, result)
    return result

#model training, also adapted from above tutorial
def GCN_train(spec_dig_graph, model):
  optimizer = tch.optim.Adam(model.parameters(), lr = 0.005)
  features = spec_dig_graph.ndata['pixels']

  #local variables
  best_test_acc = 0
  train_hist = []
  test_hist = []

  for e in range(200):
    # Forward
    logits = model(spec_dig_graph, features)

    # Compute prediction
    pred = logits.argmax(1)

    # Compute loss
    loss = tch.nn.functional.cross_entropy(logits[0:2007], tch.nn.functional.one_hot(tch.tensor(train_truth)).type(tch.FloatTensor))

    # Compute accuracy on training/test
    train_acc = (pred[0:2007] == tch.tensor(train_truth)).float().mean()
    test_acc = (pred[2007:9298] == tch.tensor(test_truth)).float().mean()

    # Save the best test accuracy.
    if best_test_acc < test_acc:
        best_test_acc = test_acc

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Print data and save for graph
    if e % 5 == 0:
      print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, test acc: {:.3f} (best {:.3f})'.format(e, loss, train_acc, test_acc, best_test_acc))
    train_hist.append(train_acc)
    test_hist.append(test_acc)
  
  #print graphs after loop
  plt.plot(train_hist, label = 'train acc')
  plt.plot(test_hist, label = 'test acc')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc = 'lower right')
  plt.show()

  #Confusion Matrix
  GCN_mat = np.zeros((10, 10))
  GCN_conf = pred[2007:9298]

  for i in range(len(Conv_conf)):
    GCN_mat[GCN_conf[i]][test_truth[i]] += 1

  GCN_mat_out = pd.DataFrame(GCN_mat, range(10), range(10))
  sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
  sn.heatmap(GCN_mat_out, annot = True, fmt = 'g')
  plt.xlabel('Ground Truth')
  plt.ylabel('Graph Conv. Network (test)')
  plt.title('Confusion - Ground Truth vs. GCN')
  plt.show()

  return pred

#back into main()
GCNmodel = GCN(spec_dig_graph.ndata['pixels'].shape[1], 100, 25, 10)
GCN_pred = GCN_train(spec_dig_graph, GCNmodel)

#Task 4 - Spatial-Based GNN

spat_dig_graph = dig_graph

#adapted from tutorial on DGL website
class GAT(tch.nn.Module):
  def __init__(self, in_f, h_f1, h_f2, classes):
    super(GAT, self).__init__()
    self.layer1 = dgl.nn.GATConv(in_f, h_f1, 1, feat_drop = 0.05)
    self.layer2 = dgl.nn.GATConv(h_f1, h_f2, 1, feat_drop = 0.05)
    self.layer3 = dgl.nn.GATConv(h_f2, classes, 1, feat_drop = 0.05)

  def forward(self, spat_dig_graph, in_feat):
    result = self.layer1(spat_dig_graph, in_feat)
    result = tch.nn.functional.relu(result)
    result = self.layer2(spat_dig_graph, result)
    result = tch.nn.functional.relu(result)
    result = self.layer3(spat_dig_graph, result)
    return result

#model training, also adapted from above tutorial
def GAT_train(spat_dig_graph, model):
  optimizer = tch.optim.Adam(model.parameters(), lr = 0.005)
  features = spat_dig_graph.ndata['pixels']

  #local variables
  best_test_acc = 0
  train_hist = []
  test_hist = []

  for e in range(200):
    # Forward
    logits = model(spat_dig_graph, features)
    logits = logits[0:, 0, 0, 0, 0:]

    # Compute prediction
    pred = logits.argmax(1)

    # Compute loss
    loss = tch.nn.functional.cross_entropy(logits[0:2007], tch.nn.functional.one_hot(tch.tensor(train_truth)).type(tch.FloatTensor))

    # Compute accuracy on training/test
    train_acc = (pred[0:2007] == tch.tensor(train_truth)).float().mean()
    test_acc = (pred[2007:9298] == tch.tensor(test_truth)).float().mean()

    # Save the best test accuracy.
    if best_test_acc < test_acc:
        best_test_acc = test_acc

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Print data and save for graph
    if e % 5 == 0:
      print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, test acc: {:.3f} (best {:.3f})'.format(e, loss, train_acc, test_acc, best_test_acc))
    train_hist.append(train_acc)
    test_hist.append(test_acc)

  #print graph after loop
  plt.plot(train_hist, label = 'train acc')
  plt.plot(test_hist, label = 'test acc')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc = 'lower right')
  plt.show()

  #Confusion Matrix
  GAT_mat = np.zeros((10, 10))
  GAT_conf = pred[2007:9298]

  for i in range(len(Conv_conf)):
    GAT_mat[GAT_conf[i]][test_truth[i]] += 1

  GAT_mat_out = pd.DataFrame(GAT_mat, range(10), range(10))
  sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
  sn.heatmap(GAT_mat_out, annot = True, fmt = 'g')
  plt.xlabel('Ground Truth')
  plt.ylabel('Graph Attn. Network (test)')
  plt.title('Confusion - Ground Truth vs. GAT')
  plt.show()

  return pred

#back into main()
GATmodel = GAT(spat_dig_graph.ndata['pixels'].shape[1], 100, 25, 10)
GAT_pred = GAT_train(spat_dig_graph, GATmodel)

#Results

#Deep NN vs. GCN
DeepGCN = np.zeros((10, 10))
for i in range(len(Conv_pred)):
  DeepGCN[GCN_pred[i]][Conv_pred[i]] += 1

DeepGCN_out = pd.DataFrame(DeepGCN, range(10), range(10))
sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
sn.heatmap(DeepGCN_out, annot = True, fmt = 'g')
plt.xlabel('Deep Neural Network')
plt.ylabel('Graph Conv. Network')
plt.title('Confusion - DNN vs. GCN')
plt.show()

#Deep NN vs. GAT
DeepGAT = np.zeros((10, 10))
for i in range(len(GCN_pred)):
  DeepGAT[GAT_pred[i]][Conv_pred[i]] += 1

DeepGAT_out = pd.DataFrame(DeepGAT, range(10), range(10))
sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
sn.heatmap(DeepGAT_out, annot = True, fmt = 'g')
plt.xlabel('Deep Neural Network')
plt.ylabel('Graph Attn. Network')
plt.title('Confusion - DNN vs. GAT')
plt.show()

#GCN vs. GAT
GCN_GAT = np.zeros((10, 10))
for i in range(len(GAT_pred)):
  GCN_GAT[GAT_pred[i]][GCN_pred[i]] += 1

GCN_GAT_out = pd.DataFrame(GCN_GAT, range(10), range(10))
sn.set(font_scale = 0.9, rc = {'figure.figsize':(12, 12)})
sn.heatmap(GCN_GAT_out, annot = True, fmt = 'g')
plt.xlabel('Graph Conv. Network')
plt.ylabel('Graph Attn. Network')
plt.title('Confusion - GCN vs. GAT')
plt.show()