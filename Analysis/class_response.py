import csv
import matplotlib.pyplot as plt

num_label = {'0':'Coat', '1':'Pullover', '2':'Shirt'}

class_respond_dict = {'Coat':{'Coat':0, 'Pullover':0, 'Shirt':0}, 'Pullover':{'Coat':0, 'Pullover':0, 'Shirt':0}, 'Shirt':{'Coat':0, 'Pullover':0, 'Shirt':0}}
with open('../Data/annotated_data.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        label = row[2]
        truth = num_label[row[1][0]]
        class_respond_dict[truth][label] += 1

for true_class in class_respond_dict:
    respond_dict = class_respond_dict[true_class]
    label_list = list(respond_dict.keys())
    num_list = list(respond_dict.values())
    plt.rcParams["font.size"] = 14
    plt.pie(num_list, labels=label_list, startangle=90, autopct="%1.1f%%")
    plt.savefig(f'../Results/response-{true_class}.png', bbox_inches="tight")
    plt.clf()