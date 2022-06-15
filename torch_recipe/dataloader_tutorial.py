import wave
import torch
import torchaudio


'''
한 사람이 히브리어로 Yes 혹은 No라고 녹음한 오디오 클림 60개
각 클립당 8개의 단어로 이루어져 있음
'''
yesno_data = torchaudio.datasets.YESNO('./', download=True)

n=3 ### 3번째 항목 확인
waveform, sample_rate, labels = yesno_data[n]
print(f'Waveform : {waveform}\nSample Rate : {sample_rate}\nLabels : {labels}')

'''
data loader로 불러오기
'''
data_loader = torch.utils.data.DataLoader(yesno_data, batch_size=1, shuffle=True)

'''
Data Loader로 데이터를 담았으니, 이제 얘네들을 Interation할 수 있게 됨
'''
print('++++++++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++ 다른 DATA도 확인 +++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++++++++')

for data in data_loader :
    print('DATA : ',data)
    print(f'Waveform : {data[0]}\nSample Rate : {data[1]}\nLabels : {data[2]}')
    break

'''
데이터 시각화
'''
import matplotlib.pyplot as plt
print('++++++++++++++++++++++++++++++++++++++++++++++++')
print('++++++++++++++++ DATA 시각화 +++++++++++++++++++')
print('++++++++++++++++++++++++++++++++++++++++++++++++')
print(data[0][0].numpy())

plt.figure()
plt.plot(waveform.t().numpy())
plt.show()