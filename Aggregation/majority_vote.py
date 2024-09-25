import csv
from collections import Counter

# setting
label_num = {'Coat':0, 'Pullover':1, 'Shirt':2}

respond_dict = {}
with open('../Data/annotated_data.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        task = row[1]
        label = label_num[row[2]]

        if task in respond_dict:
            respond_dict[task].append(label)
        else:
            respond_dict[task] = [label]

task_label_dict = {}
for task in respond_dict:
    respond_list = respond_dict[task]
    estimate_class = Counter(respond_list).most_common()[0][0]
    task_label_dict[task] = estimate_class

index_task_label_list = [["","img","label"]]
index = 0
for task in respond_dict:
    index_task_label_list.append([index, task, task_label_dict[task]])
    index += 1

with open('../Data/train-majority-vote.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(index_task_label_list)