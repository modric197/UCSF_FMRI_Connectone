import torch
import torch.nn.functional as F 
import pandas as pd
import numpy as np
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

cttr = random.sample(range(0, 200), 105) # randomly choose 105 datapoints of control group as training data
titr = random.sample(range(0, 196), 105) # randomly choose 105 datapoints of tinnitus group as training data

ctct = np.zeros(200)
titi = np.zeros(196)

for t in range(105):
    ctct[cttr[t]] = 1
    titi[titr[t]] = 1

# testing data
ctte = np.zeros(95, dtype = int)
tite = np.zeros(91, dtype = int)

cntc = 0
for i in range(200):
    if (ctct[i] == 0):
        ctte[cntc] = i
        cntc = cntc+1

cntt = 0
for i in range(196):
    if (titi[i] == 0):
        tite[cntt] = i
        cntt = cntt+1

features = torch.zeros(186, 30381)
for i in range(95):
    idx = ctte[i]
    print("ct", i, idx)
    cnt = 0
    srcFile = 'fMRI_connecomes/ct/ti'+str(idx)+'.csv' # the path of testing data (control group)
    datamat = pd.read_csv(srcFile, header = None)
    for j in range(246):
        for k in range(j, 246):
            features[i][cnt] = datamat[j][k]
            cnt = cnt + 1

for i in range(0, 91):
    idx = tite[i]
    print("ti", i, idx)
    cnt = 0
    srcFile = 'fMRI_connecomes/ti/ti'+str(idx)+'.csv'  # the path of testing data (tinnitus group)
    datamat = pd.read_csv(srcFile, header = None)
    for j in range(246):
        for k in range(j, 246):
            features[95+i][cnt] = datamat[j][k]
            cnt = cnt + 1

labelc = torch.zeros(95)
labelt = torch.ones(91)

tgtx = features.type(torch.FloatTensor)
tgty = torch.cat((labelc, labelt), ).type(torch.LongTensor)

testf = torch.zeros(210, 30381)
for i in range(105):
    idx = cttr[i]
    print("tgt", i, idx)
    cnt = 0
    srcFile = 'fMRI_connecomes/ct/ti'+str(idx)+'.csv' # the path of training data (control group)
    datamat = pd.read_csv(srcFile, header = None)
    for j in range(246):
        for k in range(j, 246):
            testf[i][cnt] = datamat[j][k]
            cnt = cnt + 1

for i in range(105):
    idx = titr[i]
    print("pred", i, idx)
    cnt = 0
    srcFile = 'fMRI_connecomes/ti/ti'+str(idx)+'.csv' # the path of training data (tinnitus group)
    datamat = pd.read_csv(srcFile, header = None)
    for j in range(246):
        for k in range(j, 246):
            testf[105+i][cnt] = datamat[j][k]
            cnt = cnt + 1

predc = torch.zeros(105)
predt = torch.ones(105)

x = testf.type(torch.FloatTensor)
y = torch.cat((predc, predt), ).type(torch.LongTensor)

# build up the network
class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__() 
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer
        self.out = torch.nn.Linear(n_hidden3, n_output)       # hidden layer

    def forward(self, x):
        # Forward propagation of input values
        x = F.relu(self.hidden1(x))      # activate function
        x = F.relu(self.hidden2(x))      # activate function
        x = F.relu(self.hidden3(x))      # activate function
        x = self.out(x)         
        return x


net = Net(n_feature=30381, n_hidden1=2560, n_hidden2=784, n_hidden3=10, n_output=2) 
net.to(device)

# training
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  
loss_func = torch.nn.CrossEntropyLoss()

for t in range(200):
    x1 = x.to(device)
    out = net(x1)    

    y1 = y.to(device)

    loss = loss_func(out, y1)     # compute loss

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()        

    if t % 5 == 0:
        # The maximum probability after a softmax activation function is the predicted value
        prediction = torch.max(F.softmax(out, dim=-1), 1)[1]

        pred_y = prediction.data.cpu().numpy().squeeze()
        target_y = y.data.numpy()
        accuracy = sum(pred_y == target_y) / 210.  # compute training accuracy
        print("train_acc", t, accuracy)
    
    if t % 5 == 0:
        x2 = tgtx.to(device)
        test_out = net(x2)
        target_y = tgty.data.numpy()
        prediction = torch.max(F.softmax(test_out, dim=-1), 1)[1]
        pred_y = prediction.data.cpu().numpy().squeeze()
        acc = sum(pred_y == target_y) / 186. # compute testing accuracy
        print("test_acc", t, acc)


