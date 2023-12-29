import os.path as osp
import time

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SSGConv,SGConv,GCNConv
from model import SARMAConv
import argparse
import copy
import random
acc_lists_str = []
str_lists = []
dur = []
acc_lists = []
loss_list =[]
val_acc_list =[]
test_acc_list =[]


def logging(log_name,str_lists):
     with open("data/SGC+ARMA+V2+torch+"+log_name, 'w') as f:  # 设置文件对象
        f.writelines(str_lists)


class bestSave:
    def __init__(self):
        self.best_score = None
    def stop_step(self, acc, model, args):
        score = acc
        if self.best_score is None:
            self.best_score = score
        elif self.best_score < score:
            self.best_score = score
            torch.save({'model_state_dict': model.state_dict()}, args.dataset + "+SGC+ARMA+V2.pt")

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,num_stacks,num_layers,dropout,K):
        super().__init__()
        self.conv1 = SARMAConv(in_channels, hidden_channels, num_stacks, num_layers, KSGC=K, dropout=dropout)
        self.conv2 = SARMAConv(hidden_channels, out_channels, num_stacks, num_layers, KSGC=K,  dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):

        # x = x if self.dropout == 0 else  F.dropout(x, p = self.dropout, training=self.training)
        x = F.tanh(self.conv1(x, edge_index))
        # x = x if self.dropout == 0 else  F.dropout(x, p = self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(best_model,data,optimizer,args):
    best_Save = bestSave()
    start_time = time.time();
    for epoch in range(1,1001):
        best_model.train()
        optimizer.zero_grad()
        out = best_model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_acc, val_acc, tmp_test_acc = test(best_model, data)
        best_Save.stop_step(val_acc, best_model, args)
        loss_list.append(str(loss.item()) + " ")
        val_acc_list.append(str(val_acc) + " ")
        test_acc_list.append(str(tmp_test_acc) + " ")
    checkpoint = torch.load(args.dataset + "+SGC+ARMA+V2.pt")
    best_model.load_state_dict(checkpoint['model_state_dict'])
    train_acc, val_acc, tmp_test_acc = test(best_model, data)
    dur.append(time.time()-start_time)
    acc_lists.append(tmp_test_acc)
    acc_lists_str.append(str(tmp_test_acc) + " ")
    print(" epoch: {:4d} | Loss {:.4f} | Accuracy {:.4f} | time:{:.4f}".format(epoch, loss.item(), val_acc,
                                                                               time.time() - start_time))
    str_lists.append("\n epoch: {:4d} | Loss {:.4f} | Accuracy {:.4f} | time:{:.4f}".format(epoch, loss.item(), val_acc,
                                                                                            time.time() - start_time))
    print("Testing...")
    print("Test accuracy {:.4f} ".format(tmp_test_acc))
    str_lists.append("\n Test accuracy: {:.4f}".format(tmp_test_acc))


def test(best_model,data):
    best_model.eval()
    out, accs = best_model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(args):
    for epoch in range(1, 21):
        if args.dataset == "cora":
            # print("Training cora...")
            dataset = 'Cora'
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
            dataset = Planetoid(path, dataset, split="public", num_train_per_class=20, num_val=500, num_test=1000,
                                transform=T.NormalizeFeatures(), )
        elif args.dataset == "citeseer":
            # print("Training citeseer...")
            dataset = 'Citeseer'
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
            dataset = Planetoid(path, dataset, split="public", num_train_per_class=20, num_val=500, num_test=1000,
                                transform=T.NormalizeFeatures(), )
        elif args.dataset == "pubmed":
            # print("Training pubmed...")
            dataset = 'Pubmed'
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
            dataset = Planetoid(path, dataset, split="public", num_train_per_class=20, num_val=500, num_test=1000,
                                transform=T.NormalizeFeatures(), )
        else:
            raise ValueError("Unknown dataset: {}".format(args.dataset))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        val_acc_list.append("\n " + str(epoch) + " ")
        test_acc_list.append("\n " + str(epoch) + " ")
        loss_list.append("\n " + str(epoch) + " ")
        data = dataset[0]
        model, data = Net(dataset.num_features,args.hidden_num,dataset.num_classes,args.num_stacks,args.num_layers,args.dropout,args.K).to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,  weight_decay=args.weight_decay)
        train(model,data,optimizer,args)
    mean = np.around(np.mean(acc_lists, axis=0), decimals=3)
    std = np.around(np.std(acc_lists, axis=0), decimals=3)
    del (dur[0])
    mean_time = np.around(np.mean(dur, axis=0), decimals=3)
    std_time = np.around(np.std(dur, axis=0), decimals=3)
    print("Total acc: ", acc_lists)
    print("mean", mean)
    print("std", std)

    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % d' % (total))
    str_lists.append("\n Total acc:")
    str_lists.extend(acc_lists_str)
    str_lists.append("\n mean:{:.4f}".format(mean))
    str_lists.append("\n std:{:.4f}".format(std))
    str_lists.append("\n mean_time:{:.4f}".format(mean_time))
    str_lists.append("\n std_time:{:.4f}".format(std_time))
    str_lists.append("\n Number of parameter: {: d}".format(total))
    log_name = args.dataset + "+" + str( args.num_stacks )+ "+" +str( args.K ) + ".txt"
    logging(log_name, str_lists)
    log_loss_name = args.dataset  + str( args.num_stacks )+ "+" +str( args.K ) + "+loss.txt"
    logging(log_loss_name, loss_list)
    log_val_name = args.dataset + str( args.num_stacks )+ "+" +str( args.K )  + "+val_acc.txt"
    logging(log_val_name, val_acc_list)
    log_test_name = args.dataset + str( args.num_stacks )+ "+" +str( args.K ) +"+test_acc.txt"
    logging(log_test_name, test_acc_list)
    val_acc_list.clear()
    test_acc_list.clear()
    acc_lists_str.clear()
    str_lists.clear()
    dur.clear()
    acc_lists.clear()
    loss_list.clear()

if __name__ == "__main__":
    """
    ARMA Model Hyperparameters
    """
    parser = argparse.ArgumentParser(description="ARMA GCN")
    parser.add_argument( "--dataset", type=str, default="Citeseer", help="Name of dataset.")
    parser.add_argument( "--gpu", type=int, default=0 )
    parser.add_argument("--shared-weights", action="store_true", default=True)
    parser.add_argument( "--num-stacks", type=int, default=2,)
    parser.add_argument( "--num-layers", type=int, default=1,)
    parser.add_argument("--dropout",type=float, default=0.75,)
    parser.add_argument("--lr",type=float, default= 0.02,)
    parser.add_argument("--K", type=int, default= 2, )
    parser.add_argument("--hidden-num", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=5e-3,)

    for i in range(3):
        if i == 3:
            args = parser.parse_args()
            args.dataset = "cora"
            args.dropout = 0.65
            args.num_layers = 1
            args.hidden_num = 16
            args.lr = 0.02
            args.weight_decay = 5e-3
            for j in range(1,2):
                args.num_stacks = j
                for K in range(8,9):
                    args.K = K
                    args.gpu = 0
                    main(args)
        elif i == 2:
            args = parser.parse_args()
            args.dataset = "citeseer"
            args.dropout = 0.75
            args.num_stacks = 1
            args.num_layers = 1
            args.hidden_num = 16
            args.lr = 0.03
            args.weight_decay = 5e-3
            for j in range(3, 5):
                args.num_stacks = j
                for K in range(8, 9):
                    args.K = K
                    args.gpu = 0
                    main(args)
        elif i == 3:
            args = parser.parse_args()
            args.dataset = "pubmed"
            args.dropout = 0.7
            args.num_stacks = 1
            args.num_layers = 1
            args.hidden_num = 16
            args.lr = 0.02
            args.weight_decay=5e-4
            for j in range(1, 2):
                args.num_stacks = j
                for K in range(11, 12):
                    args.K = K
                    args.gpu = 0
                    main(args)








