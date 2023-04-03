from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
#script ami kilapítja a fájlokat és mögé eredémnyben hogy hány lépés
#Pálya kilapítva: 0-1-esek 0-ra lehet rakni, 1-re nem, hány lépésből --> ebből becsüljük meg hogy hány lépésből rakható ki

# A tanulóhalmaz létrehozása
#x_train = np.random.rand(100, 56) #7x8-as pálya a legnagyobb, a kisebbre levágjuk úgy,hogy 1-esekkel feltöltjük
#y_train = np.random.rand(100) #predikció

# CSV fájl beolvasása
data = pd.read_csv('data.csv')

# Az x_train változó létrehozása
x_train = data.iloc[:, 0:56].values

# Az y_train változó létrehozása
y_train = data.iloc[:, 56].values

# A neurális háló létrehozása
model = Sequential() #Minden réteg csak a szomszédjaival van összekötve
model.add(Dense(32, input_dim=56, activation='relu')) #bemeneti réteg 32 neuronnal
model.add(Dense(16, activation='relu')) #rejtett réteg 16 neuronnal
model.add(Dense(1, activation='linear')) #kimeneti réteg 1 neuronnal

# A neurális háló veszteségfüggvénye
model.compile(loss='mean_squared_error', optimizer='adam') #négyzetes hibával

# A neurális háló tanítása
model.fit(x_train, y_train, epochs=50)  #Az epochs -csal mondjuk meg hogy mennyiszer haladjon át a tanulóhalmazon

# Egy új pálya nehézségének meghatározása
new_board = np.random.randint(2, size=(7, 8)) # Új pálya generálása
new_board = new_board.ravel() #Pálya kilapítása
difficulty = model.predict(np.array([new_board]))[0][0] #létrehozzuk a pályát és visszaadjuk a nehézségét
print(f'A pálya nehézsége: {difficulty*100:.2f}%')