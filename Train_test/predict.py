import csv
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from cnn import vgg16
from dataset import get_data

def predict(device, learner, test_data):
    pre_truth_list = []
    for data, target in test_data:
        data = data.to(device)
        y = learner(data)
        m = nn.Softmax(dim=1)
        y = m(y)
        
        pre = np.argmax(y.to('cpu').numpy().flatten())
        truth = target.numpy()[0]
        pre_truth_list.append([pre, truth])
    
    with open('../Results/vgg-prediction.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(pre_truth_list)

def main(learner):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner.load_state_dict(torch.load("./Params/vgg.pth"))
    train_data, test_data = get_data(batch_size=1) 
    
    learner = learner.to(device)
    cudnn.benchmark = True

    with torch.no_grad():
        predict(device, learner, test_data)
        
if __name__ == "__main__":
    learner = vgg16()
    print("Start Training")
    print("")
    main(learner)