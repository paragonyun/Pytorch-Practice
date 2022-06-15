import torch
import math

class Polynomial3(torch.nn.Module) :
    def __init__(self) :
        super().__init__()
        '''
        사용할 변수 정의
        '''
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
    
    ## forward 정의  (model.forward()하면 return되는 것 정의)
    def forward(self, x) :
        return self.a + self.b*x + self.c*x**2 + self.c*x**3 ## 여기 ^로 하고 **로 안 하면 에러뜸

    def print_part(self) :
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = Polynomial3()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    if t % 100 == 0 :
        print(f"|EPOCH {t} | {loss}|")

    optimizer.zero_grad() ## model.zero_grad와 optimizer.zero_grad 차이 알기

    loss.backward()

    optimizer.step()    

print(f'모델의 추정 식\n{model.print_part()}')