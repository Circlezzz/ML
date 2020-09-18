from sklearn.datasets import load_boston
import pandas as pd

bos = load_boston()
print(bos.keys())

df = pd.DataFrame(bos.data)
df.columns = bos.feature_names
df['Price'] = bos.target
print(df.head())
print(df.describe())


data = df[df.columns[:-1]]
data = data.apply(
    lambda x: (x - x.mean()) / x.std()
)

data['Price'] = df.Price

import numpy as np

X = data.drop('Price', axis=1).to_numpy()
Y = data['Price'].to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

import torch

print(torch.__version__)

n_train = X_train.shape[0]
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
Y_train = torch.tensor(Y_train, dtype=torch.float).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float).view(-1, 1)

w_num = X_train.shape[1]
net = torch.nn.Sequential(
    torch.nn.Linear(w_num, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1)
)

torch.nn.init.normal_(net[0].weight, mean=0, std=0.1)
torch.nn.init.constant_(net[0].bias, val=0)

datasets = torch.utils.data.TensorDataset(X_train, Y_train)

train_iter = torch.utils.data.DataLoader(datasets, batch_size=10, shuffle=True)

loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    for x, y in train_iter:
        output = net(x)
        l = loss(output, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch {} loss: {:.4f}".format(epoch + 1, l.item()))

print('MSE:'+str(loss(net(X_test), Y_test).item()))

print('Ground Truth:'+str(Y_test[11:20]))
print('Predicted:'+str(net(X_test[11:20]).data))
