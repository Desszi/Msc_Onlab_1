import csv
import math

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#Az adatok be (x) és kimenetének (y) implementálása
x_data = []
y_data = []

with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array_string = row[0]
        number = int(row[1])
        matrix = np.array([[int(cell) for cell in line.split(',')] for line in array_string.split('\n')])
        x_data.append(matrix)
        y_data.append(number)
        #Az elforgatottjai is a mátrixnak a beolvasott adatok közé kerüljön
        for i in range(1, 4):
            rotated_matrix = np.rot90(matrix, i)
            x_data.append(rotated_matrix)
            y_data.append(number)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2) #tanuló és teszthalmaz lebontás 80-20%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) #tanuló és validációshalmaz lebontás 80-20%

# A neurális háló létrehozása
# Minden réteg csak a szomszédjaival van összekötve
model = Sequential()
model.add(Dense(32, input_dim= matrix.size, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# A neurális háló veszteségfüggvénye
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001)) #négyzetes hibával

# A neurális háló tanítása
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape((x_train.shape[0], matrix.size))
x_val = x_val.reshape((x_val.shape[0], matrix.size))
x_test = x_test.reshape((x_test.shape[0], matrix.size))

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, callbacks=[early_stopping])
loss = model.evaluate(x_test, y_test)

# Egy új pálya nehézségének meghatározása
#new_board = np.random.randint(2, size=(int(math.sqrt(matrix.size)), int(math.sqrt(matrix.size))))# Új pálya generálása

new_board = [
[0,0,0,0,1,1],
[1,1,0,0,1,0],
[0,0,0,0,1,0],
[0,0,0,0,0,0],
[0,0,0,0,0,0],
[0,0,0,0,0,0] ]

new_board = np.array(new_board)
print("A jelenlegi pálya:", new_board)
new_board = new_board.ravel() #Pálya kilapítása
steps = model.predict(np.array([new_board]))[0][0] #létrehozzuk a pályát és visszaadjuk a nehézségét
print(f'A pálya várható lépésszáma: {round(steps)}')