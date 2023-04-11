import csv
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

# CSV fájl beolvasása
data = pd.read_csv('data.csv')

#A tanulóhalmaz implementálása
x_train = []
y_train = []

with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array_string = row[0]
        number = int(row[1])
        matrix = [[int(cell) for cell in line.split(',')] for line in array_string.split('\n')]
        x_train.append(matrix)
        y_train.append(number)

# A neurális háló létrehozása
model = Sequential() #Minden réteg csak a szomszédjaival van összekötve
model.add(Dense(32, input_dim=36, activation='relu')) #bemeneti réteg 32 neuronnal
model.add(Dense(16, activation='relu')) #rejtett réteg 16 neuronnal
model.add(Dense(1, activation='linear')) #kimeneti réteg 1 neuronnal

# A neurális háló veszteségfüggvénye
model.compile(loss='mean_squared_error', optimizer='adam') #négyzetes hibával

# A neurális háló tanítása
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], 36))
model.fit(x_train, y_train, epochs=300)  #Az epochs -csal mondjuk meg hogy mennyiszer haladjon át a tanulóhalmazon

# Egy új pálya nehézségének meghatározása
#new_board = np.random.randint(2, size=(6, 6)) # Új pálya generálása
new_board = [
[0,0,0,0,0,1],
[0,0,0,0,0,0],
[0,0,1,1,0,0],
[0,0,0,0,0,1],
[0,0,0,0,0,1],
[0,0,0,0,0,1]]

new_board = np.array(new_board)
print("A jelenlegi pálya:", new_board)
new_board = new_board.ravel() #Pálya kilapítása
steps = model.predict(np.array([new_board]))[0][0] #létrehozzuk a pályát és visszaadjuk a nehézségét
print(f'A pálya várható lépésszáma: {round(steps)}')