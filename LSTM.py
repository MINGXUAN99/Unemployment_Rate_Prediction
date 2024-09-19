import matplotlib.pyplot as plt

from _init_ import *
import torch
from torch import nn


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size = 1, output_size = 1, num_layers = 1, ):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)   
        self.linear = nn.Linear(hidden_size, output_size)   

    def forward(self, _x):
        x, _ = self.lstm(_x) # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x
    
def lossFunction():
    return nn.MSELoss()


def train(model, max_epochs = MAX_EPOCHS):
    max_epochs = MAX_EPOCHS
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)

    model.train()
    for epoch in range(max_epochs):
        output = model(train_x)
        loss = lossFunction(output, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 1e-3:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    return model

def evaluation(model):
    pred_y_train = model(train_x).to(device)
    pred_y_train = pred_y.cpu().view(-1, OUTPUT_FEATURES).data.numpy()

    model = model.eval()  # switch to testing model

    test_x = test_x.reshape(-1, 1, INPUT_FEATURES)
    test_x = torch.from_numpy(test_x)
    test_x = test_x.to(device)

    pred_y_test = model(test_x)
    pred_y_test = pred_y_test.cpu().view(-1, OUTPUT_FEATURES).data.numpy()

    loss = lossFunction(torch.from_numpy(pred_y), torch.from_numpy(test_y))
    print("test RMSE:", np.sqrt(loss.item()))
    return pred_y_train, pred_y_test

def plot(outcome, train_y = train_y):
    pred_y_for_train, pred_y_for_test = evaluation
    plt.figure()
    plt.plot(range(len(train_y)), train_y, 'b', label='y_trn')
    plt.plot(range(len(train_y)), pred_y_for_train, 'y--', label='pre_trn')

    plt.plot(range(len(train_y),len(train_y)+len(test_y)), test_y, 'k', label='y_tst')
    plt.plot(range(len(train_y),len(train_y)+len(test_y)), pred_y_for_test, 'm--', label='pre_tst')

    plt.xlabel('t')
    plt.ylabel('Rate')
    plt.show()


model = LSTM_model(INPUT_FEATURES, 20, OUTPUT_FEATURES, 1)
model = train(model)
outcome = evaluation(model)
plot(outcome)