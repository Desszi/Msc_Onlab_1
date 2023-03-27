from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# A tanulóhalmaz létrehozása
x_train = np.random.rand(1000, 25) #5x5-ös pálya
y_train = np.random.rand(1000) #predikció
print(x_train)

# A neurális háló létrehozása
model = Sequential() #Minden réteg csak a szomszédjaival van összekötve
model.add(Dense(16, input_dim=25, activation='relu')) #bemeneti réteg 16 neuronnal
model.add(Dense(8, activation='relu')) #rejtett réteg 6 neuronnal
model.add(Dense(1, activation='sigmoid')) #kimeneti réteg 1 neuronnal

# A neurális háló veszteségfüggvénye
model.compile(loss='mean_squared_error', optimizer='adam') #négyzetes hibával

# A neurális háló tanítása
model.fit(x_train, y_train, epochs=100)  #Az epochs -csal mondjuk meg hogy mennyiszer haladjon át a tanulóhalmazon

# Egy új pálya nehézségének meghatározása
new_board = np.random.rand(25) #random pálya hogy valaminek a nehézségét vizsgáljunk
difficulty = model.predict(np.array([new_board]))[0][0] #létrehozzuk a pályát és visszaadjuk a nehézségét
print(f'A pálya nehézsége: {difficulty*100:.2f}%')