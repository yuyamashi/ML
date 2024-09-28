import csv
from argparse import ArgumentParser as ArgPar
import datetime
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from cnn import vgg16
from dataset import get_data


train_acc_list = []
test_acc_list = []


def print_result(values):
    f1 = "  {"
    f2 = "}  "
    f_int = "{}:<20"
    f_float = "{}:<20.5f"
    f_vars = ""
    
    for i, v in enumerate(values):
        if type(v) == float:
            f_vars += f1 + f_float.format(i) + f2
        else:
            f_vars += f1 + f_int.format(i) + f2
    
    print(f_vars.format(*values))


def accuracy(y, target):
    pred = y.data.max(1, keepdim = True)[1]
    acc = pred.eq(target.data.view_as(pred)).cpu().sum()

    return acc


def train(device, optimizer, learner, train_data, loss_func):
    train_acc, train_loss, n_train = 0, 0, 0
    lr = optimizer.param_groups[0]["lr"]
    learner.train()
    bar = tqdm(desc = "Training", total = len(train_data), leave = False)
    
    for data, target in train_data:
        data, target =  data.to(device), target.to(device)
        y = learner(data)
        loss = loss_func(y, target)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        train_acc += accuracy(y, target)
        train_loss += loss.item() * target.size(0)
        n_train += target.size(0)

        bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(train_loss / n_train, float(train_acc) / n_train))
        bar.update()
    bar.close()

    return float(train_acc) / n_train, train_loss / n_train


def test(device, learner, test_data, loss_func):
    test_acc, test_loss, n_test = 0, 0, 0
    bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
    
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        y = learner(data)
        loss = loss_func(y, target)

        test_acc += accuracy(y, target)
        test_loss += loss.item() * target.size(0)
        n_test += target.size(0)

        bar.update()
    bar.close()
    
    return float(test_acc) / n_test, test_loss / n_test


def main(learner):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data = get_data(batch_size=15) 
    
    global now
    now = datetime.datetime.now()
    
    learner = learner.to(device)
    cudnn.benchmark = True

    #optimizer = optim.Adam(learner.parameters(), lr=0.001, weight_decay=0.0005)
    optimizer = optim.SGD( \
                        learner.parameters(), \
                        lr = 1.0e-3, \
                        momentum = 0.9, \
                        weight_decay = 5.0e-4, \
                        nesterov = True \
                        )
    loss_func = nn.CrossEntropyLoss().cuda()

    rsl_keys = ["lr", "epoch", "TrainAcc", "TrainLoss", "TestAcc", "TestLoss", "Time"]
    rsl = []
    print_result(rsl_keys)
    y_out = 1.0e+8
    
    for epoch in range(50):
        lr = optimizer.param_groups[0]["lr"]
        train_acc, train_loss = train(device, optimizer, learner, train_data, loss_func)
        train_acc_list.append(train_acc)

        learner.eval()

        with torch.no_grad():
            test_acc, test_loss = test(device, learner, test_data, loss_func)
            print(test_acc)
            torch.save(learner.state_dict(), './Params/vgg.pth')
    
        time_now = str(datetime.datetime.today())
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
     
        y_out = min(y_out, test_loss)
        print_result(rsl[-1].values())

        
if __name__ == "__main__":
    learner = vgg16()
    print("Start Training")
    print("")
    main(learner)