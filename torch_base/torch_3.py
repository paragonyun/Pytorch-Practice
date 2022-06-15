import os
from re import A 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''
torch의 모든 모델들은 nn.Module의 하위 클래스임 !!!!!
때문에 처음 만들 때 이를 상속 받고 시작해야함
'''

class NeuralNetwork(nn.Module) :
    def __init__(self) :
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512), ## input output이 한번에 입력되어야함
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x) :
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


'''
여기부턴 예측 (SoftMax)
'''
model = NeuralNetwork()

X = torch.rand(1,28,28)
logits = model(X)
pred_proba = nn.Softmax(dim=1)(logits) ## 여긴 keras랑 똑같음
y_pred = pred_proba.argmax(1)
print(f'예상 클래스 : {y_pred}') ## 1이 나오는 것을 볼 수 있음

####################################################################
"""위에 class로 모아둔 게 아래에 있는 거 그냥 다 모은 거임 !! """
####################################################################

'''
여기부턴 모델 계층 !! Layer 내부에서 어떤 일이 일어나나 확인
'''
input_image =torch.rand(3,28,28) ## 28x28 크기의 이미지 3장

flatten = nn.Flatten()
flat_image = flatten(input_image)
print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'flatten된 이미지의 shape : {flat_image.shape}') ## shape이 (3,784)로 나오는걸 확인
print('++++++++++++++++++++++++++++++++++++++++++++++')

'''
nn.Linear = keras의 Dense
'''
layer1 = nn.Linear(in_features=28*28, out_features=20) ## input과 output 노드 수임
hidden1 = layer1(flat_image)
print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'Layer1의 output shape : {hidden1.shape}')
print('++++++++++++++++++++++++++++++++++++++++++++++')


'''
활성화 함수 ReLU
'''
print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'ReLU를 거치기 전의 hidden1\n{hidden1}\n\n')
ReLUed_hidden1 = nn.ReLU()(hidden1)
print(f'ReLU를 거친 후의 hidden1\n{ReLUed_hidden1}\n\n')
print('++++++++++++++++++++++++++++++++++++++++++++++')

'''
이 모든 걸 하나로 다 통합시켜주는 nn.Sequential. Keras랑 비슷함 결국
'''
seq_models = nn.Sequential(
    flatten, 
    layer1, ##layer1 = nn.Linear(in_features=28*28, out_features=20)
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_models(input_image)

'''
위의 seq_models의 결과로 나온 logits을 softmax에 넣어줌
'''
softmax = nn.Softmax(dim=1) ## dim=1을 해줘야 합이 1로 해줘서 확률로서 반환되게 만들어줌
pred_prebab = softmax(logits)
print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'예시 모델이 이미지 3장에 대해 내놓은 확률\n{pred_prebab}')
print('++++++++++++++++++++++++++++++++++++++++++++++')
