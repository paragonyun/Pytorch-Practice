import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


'''
Torch로 커스텀 데이터 셋을 지정할 때의 예시

'''
class CustomImageDataset(Dataset) : ## Dataset을 상속받음

    '''
    __init__은 처음 객체가 생성될 때 한 번만 실행됨. 
    초기화 해주는 곳으로, 다시 말해, 그냥 처음 뭐가 나올 지 정의해준다는 느낌임
    '''
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None) :
        self.img_labels = pd.read_csv(annotations_file, names=['file_name','label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    '''
    Image의 길이를 반환하게 해주는 곳
    '''
    def __len__(self) :
        return len(self.img_labels)

    '''
    Index를 사용할 수 있게 해주는 곳
    Index를 지정하면 그 결과로 image와 label을 반환해줌
    '''
    def __getitem__(self, idx) :
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform :
            image = self.transform(image)

        elif self.target_transform :
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample



