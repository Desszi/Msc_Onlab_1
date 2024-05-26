import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

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
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

inputs = tf.keras.Input(shape=(8, 8, 1))
conv1a = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same')(inputs)
conv1b = tf.keras.layers.Conv2D(32, (4, 4), activation='relu', padding='same')(inputs)
pool1a = tf.keras.layers.MaxPooling2D((2, 2))(conv1a)
pool1b = tf.keras.layers.MaxPooling2D((2, 2))(conv1b)
conv2a = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1a)
conv2b = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1b)
merged = tf.keras.layers.concatenate([conv2a, conv2b], axis=-1)
flatten = tf.keras.layers.Flatten()(merged)
dense1 = tf.keras.layers.Dense(64, activation='relu')(flatten)
outputs = tf.keras.layers.Dense(1, activation='linear')(dense1)  # Lineáris kimenet a regresszióhoz
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
model.summary()

y_train = np.array(y_train)
y_val = np.array(y_val)
x_train = np.array(x_train).reshape(-1, 8, 8, 1)
x_val = np.array(x_val).reshape(-1, 8, 8, 1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1000, callbacks=[early_stopping])

x_test = np.array(x_test).reshape(-1, 8, 8, 1)
y_test = np.array(y_test)
loss = model.evaluate(x_test, y_test)[0]
print(f"Loss: {loss}")

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
    reshaped_matrix = matrix.reshape(1, 8, 8, 1)
    estimated_steps = np.round(model.predict(reshaped_matrix)[0][0])
    original_steps = int(row[1])

    difference = abs(original_steps - estimated_steps)
    differences.append(difference)

    if estimated_steps > 100 and 200 > estimated_steps:
        difficulty = 'medium'
    elif estimated_steps < 100:
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
    meta_writer.writerow(["Tanitohalmaz merete", "Validacios merete", "Teszthalmaz merete", "Atlagos elteres", "Szoras"])
    meta_writer.writerow([len(x_train), len(x_val), len(estimated_data), round(mean_difference), round(std_deviation)])

# A becsült értékek előállítása
estimated_y_train = model.predict(x_train.reshape(-1, 8, 8, 1)).flatten()

# Osztályok létrehozása a becsült értékek alapján
estimated_class_1_indices = (estimated_y_train < 100)
estimated_class_2_indices = (estimated_y_train >= 100) & (estimated_y_train <= 200)
estimated_class_3_indices = (estimated_y_train > 200)

# Diagram létrehozása a becsült értékek alapján színezve, de az eredeti értékeket használva az x-tengelyen
plt.scatter(np.flatnonzero(estimated_class_1_indices), y_train[estimated_class_1_indices], c ='blue', alpha=0.5, label='Könnyű')
plt.scatter(np.flatnonzero(estimated_class_2_indices), y_train[estimated_class_2_indices], c ='orange', alpha=0.5, label='Közepes')
plt.scatter(np.flatnonzero(estimated_class_3_indices), y_train[estimated_class_3_indices], c = 'red', alpha=0.5, label='Nehéz')

# Cím és tengelyfeliratok hozzáadása
plt.title('A pályák lépéseinek száma')
plt.xlabel('Valódi lépésszám indexe')
plt.ylabel('Lépésszám')

# Jelmagyarázat hozzáadása
plt.legend(loc='upper right')

# Diagram megjelenítése
plt.show()