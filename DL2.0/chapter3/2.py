import torch,random
import numpy as np
def synthetic_data(w, b, num_examples):
    x=torch.normal(0,1,(num_examples,len(w)))
    y=x@w+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape(-1,1)

def data_iter(batch_size,features,labels):
    num=len(features)
    indices=list(range(num))
    random.shuffle(indices)
    for i in range(0,num,batch_size):
        batch_indices=torch.tensor(
            indices[i:min(i+batch_size,num)]
        )
        yield features[batch_indices],labels[batch_indices]
batch_size = 10
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)

b = torch.zeros(1, requires_grad=True)

print("w shape:", w.shape)
print("b shape:", b.shape)
def linreg(x,w,b):
    return x@w+b
def squared_loss(y_hat,y):
    return  (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params,lr,batch_size):
    with torch.no_grad:
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()
    
