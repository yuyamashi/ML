import csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

num_label = {'0':'Coat', '1':'Pullover', '2':'Shirt'}

respond_list = []
truth_list = []
with open('../Data/annotated_data.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        label = row[2]
        truth = num_label[row[1][0]]
        respond_list.append(label)
        truth_list.append(truth)

plt.rcParams["font.size"] = 14
cm = confusion_matrix(truth_list, respond_list, normalize='true', labels=["Coat", "Pullover", "Shirt"])
cm = pd.DataFrame(data=cm, index=["Coat", "Pullover", "Shirt"], 
                           columns=["Coat", "Pullover", "Shirt"])
sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Reds')
plt.xlabel('Response')
plt.ylabel('Ground Truth')
plt.savefig('../Results/response-cm.png', bbox_inches="tight")