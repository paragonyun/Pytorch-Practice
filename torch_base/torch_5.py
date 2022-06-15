'''
About Optimization
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

'''
epoch, batch_size, learning rate 등을 조정하는 과정
'''
learning_rate  = 1e-3
batch_size= 64
epochs=5

'''
1 epoch가 돌 때마다 최적화가 한번 이루어짐
1 epoch는 Train 과 Validation으로 나뉨

Loss Function
회귀 : nn.MSELoss
분류 : nn.NLLLoss / nn.CrossEntropyLoss
'''

loss_fn = nn.CrossEntropyLoss() ## Loss Function 정의

'''
Optimizer 로는 일단 SGD를 적용해봄
'''
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## 1단계 : optimizer.zero_grad()로 초기화
        ## => 해주는 이유 : 이전 epoch에서 미분의 결과값으로 저장된 값이 
        ##                  다음 epoch에 영향을 줘서 제대로 된 미분 값을 못 구하게 됨
## 2단계 : loss.backwards() 로 loss function에 대해 역전파 실시
## 3단계 : optimizer.step() 으로 가중치 조정

def train_loop(dataloader, model, loss_fn, optimizer) :
    size = len(dataloader.dataset)
    
    ## loop 시작
    for batch, (X,y) in enumerate(dataloader) :

        pred = model(X) ## 예측값
        loss = loss_fn(pred, y) ## loss 값

        ## 여기부터 역전파 !! 
        optimizer.zero_grad() ## 1단계
        loss.backward() ## 2단계
        optimizer.step() ## 3단계

        if batch %100 ==0 :
            loss, current = loss.item(), batch*len(X) 
            print(f'loss : {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn) :
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad() : ## test loop에서는 가중치 업데이트를 안 함 !!!
        for X, y in dataloader :
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct  /= size
    print(f'Test Error :\n Accuracy : {(100*correct):>0.1f}%, Avg Loss : {test_loss:>8f} \n')


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs=10

for t in range(epochs) :
    print(f'EPOCH {t+1}\n-------------------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print('DONE!!')
