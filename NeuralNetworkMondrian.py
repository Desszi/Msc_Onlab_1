import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Adatok be (x) és kimenetének (y) implementálása
x_data = []
y_data = []

with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array_string = row[0]
        number = int(row[1])
        matrix = np.array([[int(cell) for cell in line.split(',')] for line in array_string.split('\n')])
        rows, cols = matrix.shape
        if rows < 8 or cols < 8:
            matrix = np.pad(matrix, ((0, max(0, 8 - rows)), (0, max(0, 8 - cols))), constant_values=1)
        x_data.append(matrix)
        y_data.append(number)
        #Az elforgatottjai is a mátrixnak a beolvasott adatok közé kerüljön
        for i in range(1, 4):
            rotated_matrix = np.rot90(matrix, i)
            x_data.append(rotated_matrix)
            y_data.append(number)

# A tanítóhalmaz és a teszthalmaz elkészítése
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2) #tanuló és teszthalmaz lebontás 80-20%
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2) #tanuló és validációshalmaz lebontás 80-20%

# A neurális háló becsült lépésszámra
model = Sequential()
model.add(Dense(64, input_dim=matrix.size, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))  # Regression layer for step count

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mse'])

# Osztályok létrehozása
y_train_class = np.zeros((len(y_train), 3))
y_val_class = np.zeros((len(y_val), 3))
y_test_class = np.zeros((len(y_test), 3))

for i in range(len(y_train)):
    if y_train[i] < 50:
        y_train_class[i][0] = 1
    elif y_train[i] > 100:
        y_train_class[i][2] = 1
    else:
        y_train_class[i][1] = 1

for i in range(len(y_val)):
    if y_val[i] < 50:
        y_val_class[i][0] = 1
    elif y_val[i] > 100:
        y_val_class[i][2] = 1
    else:
        y_val_class[i][1] = 1

for i in range(len(y_test)):
    if y_test[i] < 50:
        y_test_class[i][0] = 1
    elif y_test[i] > 100:
        y_test_class[i][2] = 1
    else:
        y_test_class[i][1] = 1

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

early_stopping = EarlyStopping(monitor='loss', patience=10)
model.fit(x_train, y_train, validation_data=(x_val, y_val_class), epochs=500, callbacks=[early_stopping])
loss = model.evaluate(x_test, y_test_class)

# Becsült lépésszámokat hozzáadni az eredeti adatokhoz
with open('estimated_data.csv', 'r') as file:
    reader = csv.reader(file)
    estimated_data = list(reader)

new_original_data = []
differences = []

for row in estimated_data:
    array_string = row[0]
    matrix = np.array([[int(cell) for cell in line.split(',')] for line in array_string.split('\n')])
    rows, cols = matrix.shape
    if rows < 8 or cols < 8:
        matrix = np.pad(matrix, ((0, max(0, 8 - rows)), (0, max(0, 8 - cols))), constant_values=1)
    flattened_matrix = matrix.ravel()
    estimated_steps = np.round(model.predict(np.array([flattened_matrix]))[0][0])

    original_steps = int(row[1])

    difference = abs(original_steps - estimated_steps)
    differences.append(difference)

    if estimated_steps > 50 and 100 > estimated_steps:
        difficulty = 'medium'
    elif estimated_steps < 50:
        difficulty = 'easy'
    else:
        difficulty = 'hard'

    new_row = f"{array_string},{original_steps},{estimated_steps},{difficulty}"
    new_original_data.append([new_row])

# Becsült lépésszámokat írni az eredeti adatfájlba
with open('data_with_steps.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(new_original_data)

# Meta-eredmények kiszámolása
mean_difference = np.mean(differences)
std_deviation = np.std(differences)

# Meta-eredményeket írni a CSV-fájlba
with open('meta_results.csv', 'w', newline='') as meta_file:
    meta_writer = csv.writer(meta_file)
    meta_writer.writerow(["Tanitohalmaz merete", "Teszthalmaz merete", "Atlagos elteres", "Szoras"])
    meta_writer.writerow([len(x_train), len(estimated_data), mean_difference, std_deviation])

# Osztályok létrehozása
class_1 = y_train[y_train < 50]
class_2 = y_train[(y_train >= 50) & (y_train <= 100)]
class_3 = y_train[y_train > 100]

# Diagram létrehozása
plt.scatter(range(len(class_1)), class_1, c ='blue', alpha=0.5, label='Könnyű')
plt.scatter(range(len(class_2)), class_2, c ='orange', alpha=0.5, label='Közepes')
plt.scatter(range(len(class_3)), class_3, c = 'red', alpha=0.5, label='Nehéz')

# Cím és tengelyfeliratok hozzáadása
plt.title('A pályák lépéseinek száma')
plt.xlabel('Érték')
plt.ylabel('Lépésszám')
# Jelmagyarázat hozzáadása
plt.legend(loc='upper right')

# Diagram megjelenítése
plt.show()