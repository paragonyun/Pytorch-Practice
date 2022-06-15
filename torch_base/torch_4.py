'''
About Gradient for Backpropagation
'''
import torch 

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3, requires_grad=True) ## Gradient를 계산하기 위해선 이거 True로 해야됨
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x,w) + b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print('1차 loss 값 : ',loss.item()) ## loss가 보니까 dict 형태로 나와서 item()으로 봐야됨

''' 
각각의 편미분 구하기 = loss.backword()
'''
loss.backward()
print('W의 편미분 값\n', w.grad)
print('b의 편미분 값\n', b.grad)

'''
만약 가중치 업데이트를 하면 안 되는 경우 (순전파만 필요한 경우)
'''
z_det = z.detach()









