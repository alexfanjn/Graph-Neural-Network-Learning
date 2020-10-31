import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp

from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim


import math
import os



# single layer
class GCNlayer(Module):
    def __init__(self, in_features, out_features, with_bias=True):
        '''

        :param in_features: in_features size
        :param out_features: out_features size
        :param with_bias: bias item (i.e., +b?)
        '''
        super(GCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output =torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output



class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):
        super(GCN, self).__init__()
        assert device is not None,"setting device!"

        self.device = device
        self.nfeat = nfeat
        self.nhid = [nhid]
        self.nclass = nclass
        self.gc1 = GCNlayer(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GCNlayer(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        # first layer
        # whether need relu active function
        if self.with_relu:
            #self.gc1(x, adj) ---- call foward function of GCNlayer
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)

        # second layer
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)






def load_data(file_name, is_sparse=True):


    with np.load(file_name) as loader:

        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])

        else:
            adj = loader['adj_data']


        if 'attr_data' in loader:
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            features = None

        labels = loader['labels']


        if features is None:
            features = np.eye(adj.shape[0])

        features = sp.csr_matrix(features, dtype=np.float32)

        return adj, features, labels


def train_val_test(seed, labels):
    '''
    :param seed: time seed
    :param labels: label
    :return: ids of training, validation and testing set
    '''
    if seed is not None:
        np.random.seed(seed)

    # generate an numpy array form 0 to size
    idx = np.arange(len(labels))
    # size of class
    nclass = labels.max() + 1
    idx_train = []
    #contain validation and testing set
    idx_unlabeled = []

    #sample nodes based on class after permutation operation
    for i in range(nclass):
        labels_i = idx[labels==i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack((idx_unlabeled, labels_i[20: ])).astype(np.int)


    idx_unlabeled = np.random.permutation(idx_unlabeled)

    #split validation set and testing set
    idx_val = idx_unlabeled[: 500]
    idx_test = idx_unlabeled[500: 1500]

    return idx_train, idx_val, idx_test



def normalize(adj):
    '''

    :param adj: original adj
    :return: normalized adj
    '''

    # normalize to a symmetric matrix
    adj = adj + adj.T
    adj[adj > 1] = 1


    #一开始是用np.eye()，因为没有转为sp，所有最后计算DAD的时候直接内存爆炸
    # generate identity matrix
    I = sp.eye(adj.shape[0])
    # print(adj.shape[0])

    # A+I
    normalize_adj = adj + I

    # obtian degree array (with (1*size) shape)
    row_sum = np.array(adj.sum(1))

    # D^(-0.5)
    r_inv = np.power(row_sum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    # print(type(r_mat_inv))

    # D^(-0.5) * (A+I) * D^(-0.5)
    normalize_adj = normalize_adj.dot(r_mat_inv).transpose().dot(r_mat_inv)


    # print(type(normalize_adj))

    return normalize_adj


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    # load dataset
    adj, features, labels = load_data('data/cora.npz', is_sparse=True)


    # split train, validation and test set
    idx_train, idx_val, idx_test = train_val_test(2020, labels)

    # obtain D^(-0.5) * (A+I) * D^(-0.5)
    adj = normalize(adj)

    # transform to tensor type
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # transform sparse adj matrix to torch sparse tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj)




    # initialize gcn model
    model = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')

    # initialize Adam optimizer
    optimizer =optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # train epoch
    epoch = 1000
    for i in range(epoch):
        # model training
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(i + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    print("Optimization Finished!")
