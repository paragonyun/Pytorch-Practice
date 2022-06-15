import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000) ### 2000개 형성
y = torch.sin(x)

# plt.plot(x, y)
# plt.show()

p = torch.tensor([1,2,3])

## torch.unsqueeze : 특정 위치에 차원이 1 추가
## pow(p) : x^p 를 연산함
xx = x.unsqueeze(-1).pow(p) 


### nn.Sequential() = keras.models.layers.Sequential()
model = torch.nn.Sequential(
    ## torch.nn.Linear = keras.models.layers.Dense
    torch.nn.Linear(3,1),    ## input = 3 nodes, output = 1 node
    torch.nn.Flatten(0,1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
torch.optim.에서 다양한 optimize 사용 가능
'''

learning_rate = 1e-3

### optimizer의 첫번째 인자는 
### "어떤 대상이 갱신되어야 하는지" 를 지정함!!!!! 
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000) :
    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 0 :
        print(f"|EPOCH {t} | {loss}|")

    optimizer.zero_grad() ## model.zero_grad와 optimizer.zero_grad 차이 알기

    loss.backward()

    optimizer.step() ## 갱신시키는 과정 (이제 귀찮게 안 해도 됨!!)

linear_layer = model[0]

print(f'Linear로 추정한 y의 값 (Poly model)\ny={linear_layer.bias.item()}+{linear_layer.weight[:,0].item()}x + {linear_layer.weight[:,1].item()}x^2 + {linear_layer.weight[:,2].item()}x^3')