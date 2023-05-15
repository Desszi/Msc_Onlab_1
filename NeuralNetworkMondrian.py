import csv
import random

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#Az adatok be (x) és kimenetének (y) implementálása
x_data = []
y_data = []

with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array_string = row[0]
        number = int(row[1])
        matrix = np.array([[int(cell) for cell in line.split(',')] for line in array_string.split('\n')])
        rows, cols = matrix.shape
        #8x8-as mátrix létrehozása
        if rows < 8 or cols < 8:
            matrix = np.pad(matrix, ((0, max(0, 8 - rows)), (0, max(0, 8 - cols))), constant_values=1)
        x_data.append(matrix)
        y_data.append(number)
        #Az elforgatottjai is a mátrixnak a beolvasott adatok közé kerüljön
        for i in range(1, 4):
            rotated_matrix = np.rot90(matrix, i)
            x_data.append(rotated_matrix)
            y_data.append(number)

#print(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2) #tanuló és teszthalmaz lebontás 80-20%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) #tanuló és validációshalmaz lebontás 80-20%

# A neurális háló architektúrájának létrehozása
model = Sequential()
model.add(Dense(64, input_dim= matrix.size, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# A neurális háló veszteségfüggvénye
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Osztályok létrehozása
y_train_class = np.zeros((len(y_train), 3))
y_val_class = np.zeros((len(y_val), 3))
y_test_class = np.zeros((len(y_test), 3))

#Ha 100-nál kevesebb lépésből kirakható akkor könnyű, 200-nál többlépésnél nehéz és közte pedig közepes nehézségű a pálya
for i in range(len(y_train)):
    if y_train[i] < 100:
        y_train_class[i][0] = 1
    elif y_train[i] > 200:
        y_train_class[i][2] = 1
    else:
        y_train_class[i][1] = 1

for i in range(len(y_val)):
    if y_val[i] < 100:
        y_val_class[i][0] = 1
    elif y_val[i] > 200:
        y_val_class[i][2] = 1
    else:
        y_val_class[i][1] = 1

for i in range(len(y_test)):
    if y_test[i] < 100:
        y_test_class[i][0] = 1
    elif y_test[i] > 200:
        y_test_class[i][2] = 1
    else:
        y_test_class[i][1] = 1

#Adatok átformázása numpy tömbökké, valamint megfelelő formátum elérése
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape((x_train.shape[0], matrix.size))
x_val = x_val.reshape((x_val.shape[0], matrix.size))
x_test = x_test.reshape((x_test.shape[0], matrix.size))

# A neurális háló tanítása
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(x_train, y_train_class, validation_data=(x_val, y_val_class), epochs=500, callbacks=[early_stopping])
loss = model.evaluate(x_test, y_test_class)

# Véletlenszerűen kiválasztunk 10 indexet a tanítóhalmazból
random_indices = random.sample(range(len(x_train)), 10)
x_random = [x_train[i] for i in random_indices]
y_random = [y_train[i] for i in random_indices]

y_random_pred = model.predict(np.array(x_random))

for i in range(10):
    if y_random_pred[i][0] > y_random_pred[i][1] and y_random_pred[i][0] > y_random_pred[i][2]:
        print(f"A(z) {i+1}. elem nehézsége: könnyű")
    elif y_random_pred[i][1] > y_random_pred[i][0] and y_random_pred[i][1] > y_random_pred[i][2]:
        print(f"A(z) {i+1}. elem nehézsége: közepes")
    else:
        print(f"A(z) {i+1}. elem nehézsége: nehéz")

# Egy új pálya nehézségének meghatározása a tanultak alapján
#new_board = np.random.randint(2, size=(int(math.sqrt(matrix.size)), int(math.sqrt(matrix.size))))# Új pálya generálása

new_board = [
[0,0,0,0,0,0,1,1],
[0,0,0,0,0,0,1,1],
[1,1,1,1,0,0,1,1],
[0,0,0,1,0,0,1,1],
[0,0,0,0,0,0,1,1],
[0,0,0,0,0,0,1,1],
[1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1]]

new_board = np.array(new_board)
print("A jelenlegi pálya:", new_board)
new_board = new_board.ravel()
difficulty = model.predict(np.array([new_board]))[0]

if difficulty[0] > difficulty[1] and difficulty[0] > difficulty[2]:
    print('A pálya nehézsége: könnyű')
elif difficulty[1] > difficulty[0] and difficulty[1] > difficulty[2]:
    print('A pálya nehézsége: közepes')
else:
    print('A pálya nehézsége: nehéz')

# Osztályok létrehozása a diagramhoz
class_1 = y_train[y_train < 100]
class_2 = y_train[(y_train >= 100) & (y_train <= 200)]
class_3 = y_train[y_train > 200]

# Diagram létrehozása
plt.scatter(range(len(class_1)), class_1, c ='blue', alpha=0.5, label='Könnyű')
plt.scatter(range(len(class_2)), class_2, c ='orange', alpha=0.5, label='Közepes')
plt.scatter(range(len(class_3)), class_3, c = 'red', alpha=0.5, label='Nehéz')

# Cím és feliratok hozzáadása
plt.title('A pályák lépéseinek száma')
plt.xlabel('Érték')
plt.ylabel('Lépésszám')
plt.legend(loc='upper right')
plt.show()
