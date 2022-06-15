from matplotlib.cbook import flatten
import torch
import math
import matplotlib.pyplot as plt


x = torch.linspace(-math.pi, math.pi, 2000) ### 2000개 형성
y = torch.sin(x)

# plt.plot(x, y)
# plt.show()

p = torch.tensor([1,2,3])

## torch.unsqueeze : 특정 위치에 차원이 1 추가
## pow(p) : x^p 를 연산함
xx = x.unsqueeze(-1).pow(p) 

'''
여기서 부터는 nn.Module의 활용
'''

### nn.Sequential() = keras.models.layers.Sequential()
model = torch.nn.Sequential(
    ## torch.nn.Linear = keras.models.layers.Dense
    torch.nn.Linear(3,1),    ## input = 3 nodes, output = 1 node
    torch.nn.Flatten(0,1)
)

## loss function도 keras랑 비슷하게 정의 가능
loss_fn = torch.nn.MSELoss(reduction='sum')

lr = 1e-6

''''
여기서부터는 forward 단계와 backward 단계
'''

for t in range(2000) : ## 2000의 epoch를 돌면서...
    y_pred = model(xx)

    ## loss 저장
    loss = loss_fn(y_pred, y)
    ## 특정 epoch가 되면 loss 출력
    if t % 100 ==99 :
        print(t, loss.item())


    ## 역전파 단계
    model.zero_grad() ## 저장된 grad 초기화

    ## 역전파 실시
    loss.backward()
    
    with torch.no_grad() :
        for param in model.parameters():
            param -= lr*param.grad

## 첫번째 계층만 대표적으로 한번 보겟음
linear_layer = model[0]

print(f'Linear로 추정한 y의 값 (Poly model)\ny={linear_layer.bias.item()}+{linear_layer.weight[:,0].item()}x + {linear_layer.weight[:,1].item()}x^2 + {linear_layer.weight[:,2].item()}x^3')

