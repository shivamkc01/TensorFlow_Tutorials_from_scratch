import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# loading our datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# for each time step you're going to sort of unroll
# one row of the image at a time
# let's say the first time step
# it's going to take the first row of the image and send that in
# and then for the second time step 
# it's going to take the second row and send that in...

# Just remember you wouldn't use sequence models to handle images
# it is not the best model for it you would use a convnet, rnn, lstm, gru.




""" Simple RNN Model """
model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))  # Here None means we are not specifying time steps, 28 pixels at each time step.
model.add(
    layers.SimpleRNN(512, return_sequences=True, activation='relu')  # it is returning the output from each time step and in that way we can stack multiple rnns layers on top of each other
    # output of this rnn is 512 nodes -> (None, None, 512) 
)
model.add(layers.SimpleRNN(512, activation = 'relu'))   # (None, None, 512) -> first None is for the batches and second one is for the hidden states.
model.add(layers.Dense(10))       # (None, 10)
print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(x_train,y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)




""" Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks """

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.GRU(256, return_sequences=True, activation='tanh')
)
model.add(layers.GRU(256, activation='tanh'))
model.add(layers.Dense(10))
print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)




""" Long short-term memory (LSTM) """

model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
model.add(
    layers.LSTM(256, return_sequences=True, activation="relu")
)
model.add(layers.LSTM(256, name="lstm_layer2", activation='relu'))
model.add(layers.Dense(10))

model.compile(
	loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer = keras.optimizers.Adam(leraning_rate=0.001))
model.fit(x_train, y_train, batch_size = 64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, epochs=2)
