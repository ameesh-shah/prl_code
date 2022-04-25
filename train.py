import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 128
EPOCHS = 40

class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.X = torch.from_numpy(x_data)
        self.Y = torch.from_numpy(y_data)
        self.len = self.X.shape[0]

    def __getitem__(self,index): 
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len

def load_data(outfilepath):
    statefilepath = "{}_states".format(outfilepath)
    actionfilepath = "{}_actions".format(outfilepath)
    state_data = np.load(statefilepath)
    action_data = np.load(actionfilepath)
    print(state_data[0])
    print(action_data[0])
    state_train, state_test, action_train, action_test = train_test_split(state_data, action_data, test_size=0.2, random_state=42)
    train_dataset = Data(state_train, action_train)
    test_dataset = Data(state_test, action_test)
    print("finished loading data")
    return train_dataset, test_dataset

def run_train(outfilepath):
    train_dataset, test_dataset = load_data(outfilepath)
    train_loader = torch.utils.data.DataLoader(train_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    model = nn.Sequential(
        nn.Linear(115, 200),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(200, 150),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(150, 100),
        nn.LayerNorm(100),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.Dropout(0.1),
        nn.ReLU(),
        nn.Linear(50, 8),
    )

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS+1):
        print('Epoch {}/{}'.format(epoch, EPOCHS))
        train_test_model(model, loss_function, optimizer, train_loader, True)
    
    train_test_model(model, loss_function, optimizer, test_loader, False)
    return model

def train_test_model(model, loss_function, optimizer, data_loader, train):
    if train:
        model.train()
    else:
        model.eval()

    current_loss = 0.0
    current_error = 0

    for i, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs.float())
            loss = loss_function(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        current_loss += loss.item() * inputs.size(0)
        current_error += torch.sqrt(torch.sum(torch.square(outputs - labels.data))/len(outputs))

    total_loss = current_loss / len(data_loader.dataset)
    total_error = current_error.double() / len(data_loader.dataset)

    if train:
        print('Train Loss: {:.4f}; Error: {:.4f}'.format(total_loss, total_error))
    else:
        print('Test Loss: {:.4f}; Error: {:.4f}'.format(total_loss, total_error))

#model = run_train()