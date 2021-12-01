import torch
from torch import nn
import matplotlib.pyplot as plt
import data_preprocessing as data 
from models import MLP
from torch.utils.tensorboard import SummaryWriter
import datetime
#HYPERPARAMETERS 

BATCH_SIZE = 32
n_epoch = 100
lr = 0.0001
dropout = 0.2
dim_hidden1 = 32 
dim_hidden2 = 2 

train, test = data.load_dataset(BATCH_SIZE)

input, label = next(iter(train))
input_dim = input.shape[1] #Number of input features 
output_dim = 1 #Number of outputs to give out 

writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

model = MLP(input_dim,dim_hidden1,dim_hidden2,output_dim)
criterion = nn.MSELoss()
optim = torch.optim.SGD(params = model.parameters(), lr=lr)

avg_training_loss = []
avg_testing_loss = []
for i in range(n_epoch):
    epoch_loss_train = []
    epoch_loss_test = []
    epoch_avg_acc = []
    for x,y in train : 
        optim.zero_grad()
        yhat = model(x.float())
        loss = criterion(yhat,y.float())
        loss.backward()
        optim.step() 
        epoch_loss_train.append(loss.item()) #for each batch, we add the loss to a tab
    avg_train_loss = sum(epoch_loss_train)/len(epoch_loss_train) #that allows us to compute the avg loss of the epoch
    avg_training_loss.append(avg_train_loss) #that we keep for further tracking throughout learning
    writer.add_scalar(f'Training loss',avg_train_loss,i)

    with torch.no_grad():
        for x_test,y_test in test : 
            batch_acc = 0
            yhat = model(x_test.float())
            test_loss = criterion(yhat,y_test.float())
            epoch_loss_test.append(test_loss)
    
    avg_test_loss = sum(epoch_loss_test)/len(epoch_loss_test) 
    avg_testing_loss.append(avg_test_loss) 
    
    writer.add_scalar(f'Testing loss',avg_test_loss,i)

epochs = range(n_epoch)
fig = plt.figure(1) 
plt.plot(epochs,avg_training_loss,'g', label ="Training loss")
plt.plot(epochs,avg_testing_loss,'b',label="Testing loss")
plt.legend()
plt.show()