import numpy as np
from scipy.spatial import distance
import warnings
import pandas as pd
import random
# import matplotlib.pyplot as plt
# from matplotlib import style
from collections import Counter
# style.use('fivethirtyeight')

dataset = {'k' : [[1, 2], [2,3], [3,1]], 'r' : [[6, 5], [7, 7], [8, 6]]}
new_feature = [5, 7]

def k_nearest_neighbors(data, predict, k=3):
    if(len(data) >= k):
        warnings.warn('invalid value of k')
    
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = distance.euclidean(features, predict)
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    votes_res = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    # print(votes_res, confidence)

    return votes_res, confidence
accuracies = []
for i in range(5):
    df = pd.read_csv('C:/Users/Lenovo/Desktop/Machine learning Course/From Youtube/K Nearest Neighbors/breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = .4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total   = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if(group == vote):
                correct += 1

            total += 1

    accuracy = correct/total
    print('accuracy', accuracy)
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))