from sklearn.neural_network import MLPClassifier

raw_x = [18023140, 18023052, 18023144]
x = [0 for i in range(len(raw_x))]

for i in range(len(raw_x)):
    x[i] = [int(i) for i in list(str(raw_x[i]))]
y = [38916, 37641, 38768]

clf = MLPClassifier(hidden_layer_sizes=[8], activation='logistic', solver='sgd', random_state=3)
clf.fit(x, y)
for xx in x:
    print('0' + str(clf.predict([xx])[0]))
