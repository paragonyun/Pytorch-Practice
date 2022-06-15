import torch
import torchvision.models as models

model = models.vgg16(pretrained=True) ## 이미 사전학습된 vgg16을 가져옴

## weight들은 state_dict에 저장되어 있음. 
## 그것들을 torch.save로 저장
torch.save(model.state_dict(), 'model_weights.pth') 

'''
저장된 weight를 불러오는 방법
'''
model = models.vgg16() ## 그냥 모델을 불러옴. pretarined=True를 안 해줬음!!
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() ## 모델을 평가모드로 바꿈

'''
모델 저장 방법
'''
torch.save(model, 'model.pth')

'''
모델 불러오기
'''
model = torch.load('model.pth')



