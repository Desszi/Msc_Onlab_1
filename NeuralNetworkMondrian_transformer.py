import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.layers import MultiHeadAttention, LayerNormalization, Reshape, GlobalAveragePooling1D, concatenate
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

x_data, y_data = [], []
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array_string = row[0]
        number = int(row[1])
        matrix = np.array([[int(cell) for cell in line.split(',')] for line in array_string.split('\n')])
        matrix = np.pad(matrix, ((0, max(0, 8 - matrix.shape[0])), (0, max(0, 8 - matrix.shape[1]))), constant_values=1)
        for i in range(4):
            x_data.append(np.rot90(matrix, i))
            y_data.append(number)
        for flip in [np.fliplr, np.flipud]:
            x_data.append(flip(matrix))
            y_data.append(number)

x_data = np.array(x_data).reshape(-1, 8, 8, 1)
y_data = np.array(y_data).reshape(-1, 1)

y_data = np.log1p(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

inputs = Input(shape=(8, 8, 1))
x_patch = tf.image.extract_patches(
    images=inputs,
    sizes=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    rates=[1, 1, 1, 1],
    padding='VALID'
)
x_patch = Reshape((16, 4))(x_patch)
x_patch = Dense(128)(x_patch)

x_trans = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256)(x_patch)
x_trans = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256)(x_trans)
x_trans = GlobalAveragePooling1D()(x_trans)

x_cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x_cnn = Conv2D(128, (3, 3), activation='relu', padding='same')(x_cnn)
x_cnn = MaxPooling2D((2, 2))(x_cnn)
x_cnn = Flatten()(x_cnn)

x = concatenate([x_trans, x_cnn])
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='linear')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=tf.keras.losses.Huber(), metrics=['mae'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=400, callbacks=[early_stopping])

y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log).flatten()
y_true = np.expm1(y_test).flatten()

easy = y_pred < 10000
medium = (y_pred >= 10000) & (y_pred <= 40000)
hard = y_pred > 40000

plt.scatter(np.flatnonzero(easy), y_true[easy], c='blue', alpha=0.5, label='Könnyű')
plt.scatter(np.flatnonzero(medium), y_true[medium], c='orange', alpha=0.5, label='Közepes')
plt.scatter(np.flatnonzero(hard), y_true[hard], c='red', alpha=0.5, label='Nehéz')
plt.title('Transformer+CNN becslések nehézségi szint szerint (8x8 pályák)')
plt.xlabel('Példány indexe')
plt.ylabel('Valódi lépésszám')
plt.legend()
plt.show()

mae = mean_absolute_error(y_true, y_pred)
print(f"Átlagos abszolút hiba (MAE): {mae:.2f}")
print(f"Szórás a különbségekben: {np.std(np.abs(y_true - y_pred)):.2f}")
