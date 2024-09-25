import csv

count = 0
with open('../Data/train-majority-vote.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        index, img, label = row
        if img[0] == label:
            count += 1
            
print(f'Majority Vote : {(count/1500)*100}%')