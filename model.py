import torch
import torch.nn as nn
import torch.nn.functional as F
class CatDogModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 16, 
                               kernel_size = 3,
                               padding=1)
        
        self.conv2 = nn.Conv2d(in_channels = 16,
                               out_channels = 32,
                               kernel_size = 3,
                               padding = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        
        self.fc2 = nn.Linear(128,2)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # 2차원 이미지를 1차원 막대기로 평탄화 (투표를 위해 한 줄로 세움)
        x = torch.flatten(x, 1)
        
        # 판결 부서 통과
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 최종 출력
        
        return x
    
    # model.py 맨 아래에 추가
if __name__ == "__main__":
    # 1. 뇌(Model) 생성
    model = CatDogModel()
    
    # 2. 가짜 트럭(가짜 데이터) 생성 
    # [배치 사이즈 8, 채널 3(RGB), 가로 224, 세로 224] 크기의 빈 박스
    dummy_input = torch.randn(8, 3, 224, 224)
    
    # 3. 뇌에 가짜 데이터를 통과시켜 보기 (Forward Pass)
    print("가짜 데이터를 뇌에 주입합니다...")
    output = model(dummy_input)
    
    # 4. 결과 확인
    print(f"최종 출력 형태: {output.shape}") 
    # 정상이라면 [8, 2]가 나와야 함 (8장의 사진에 대해 각각 2개(개/고양이)의 확률값 도출)