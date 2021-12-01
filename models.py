from torch import nn

class MLP(nn.Module):
    def __init__(self,input_dim,dim_hidden1, dim_hidden2,output_dim):
        super().__init__()
        self.dim_n = input_dim
        self.dim_h1 = dim_hidden1
        self.dim_h2 = dim_hidden2
        self.dim_o = output_dim

        self.fc1 = nn.Linear(self.dim_n,self.dim_h1)
        self.act1 = nn.Tanh()

        self.fc2 = nn.Linear(self.dim_h1,self.dim_h2)
        self.act2 = nn.Sigmoid()

        self.fc3 = nn.Linear(self.dim_h2,self.dim_o)
        self.act3 = nn.Sigmoid()
    
    def forward(self,x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.act3(self.fc3(x))


