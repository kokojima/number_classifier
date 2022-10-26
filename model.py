# 以下を「model.py」に書き込み
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
n_class = len(classes)
img_size = 28

# 画像認識のモデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(img_size*img_size, 1024)  # 全結合層
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()  # ReLU 学習するパラメータがないので使い回しできる

    def forward(self, x):
        x = x.view(-1, img_size*img_size)  # (バッチサイズ, 入力の数): 画像を1次元に変換
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

# 訓練済みパラメータの読み込みと設定
net.load_state_dict(torch.load(
    "model_num_cnn.pth", map_location=torch.device("cpu")
    ))
    
def predict(img):
    # モデルへの入力
    img = img.convert("L")  # モノクロに変換
    img = img.resize((img_size, img_size))  # サイズを変換
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.0), (1.0))
                                    ])
    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size)
    
    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
