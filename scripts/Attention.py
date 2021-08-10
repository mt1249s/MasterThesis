import numpy as np
from Bio import SeqIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
import keras
from keras.models import load_model
from matplotlib import pyplot as plt
import positional_embedding
# import sklearn
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

fasta_train = 'H_train.fasta'
csv_train = 'H_train.csv'
fasta_test = 'H_test.fasta'
csv_test = 'H_test.csv'

tf.keras.backend.clear_session
# tf.debugging.set_log_device_placement(True)

## Add label to the train dataset and generate X_train=record.seq and Y_train=label
size_train = 0
train_lst = []
train_samples = []
train_labels = []

letters = 'XACGT'
emb_dict = {letter: number for number, letter in
            enumerate(letters)}  # number+1 for emb bec
# train
with open(fasta_train)as fn:
    for record in SeqIO.parse(fn, 'fasta'):
        label_train = 0 if 'CDS' in record.id else 1
        # print(label_train)
        train_sample = []
        size_train = size_train + 1
        lst = [record.id, str(record.seq), len(record), label_train]
        # print(lst)

        for index, letter, in enumerate(record.seq):
            train_sample.append(emb_dict[letter])

        train_lst.append(lst)
        train_labels.append(label_train)
        train_samples.append(train_sample)

# padding train
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    train_samples, padding="post"
)
padded_inputs, train_labels = np.array(padded_inputs), np.array(train_labels)

# test
size_test = 0
test_lst = []
test_samples = []
test_labels = []
with open(fasta_test)as fn:
    for record in SeqIO.parse(fn, 'fasta'):
        label_test = 0 if 'CDS' in record.id else 1
        # print(label_test)
        test_sample = []
        size_test = size_test + 1
        lst = [record.id, str(record.seq), label_test]

        for index, letter, in enumerate(record.seq):
            test_sample.append(emb_dict[letter])

        test_labels.append(label_test)
        test_samples.append(test_sample)

# padding
padded_tests = tf.keras.preprocessing.sequence.pad_sequences(list(test_samples), maxlen=3000)
padded_tests, test_labels = np.array(padded_tests), np.array(test_labels)

### what are the max lengths of the train dataset?
# max_train = df_train['length'].max()
max_train = 3000

input_layer1 = layers.Input(shape=(max_train,))
embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2, input_length=3000, mask_zero=True)(input_layer1)
#flatt_output = layers.Flatten()(embedding_layer)

units = 3

query0 = layers.Dense(units, name='query_layer0')(embedding_layer)
key0 = layers.Dense(units, name='key_layer0')(embedding_layer)
values0 = layers.Dense(units, name='values_layer0')(embedding_layer)
tf.matmul(
    query0,
    key0,
    transpose_a=False,
    transpose_b=True,
    name='query_key_layer0'
)

#attention_kernel0 = layers.Attention()([query0, values0])

query1 = layers.Dense(units, name='query_layer1')(embedding_layer)
values1 = layers.Dense(units, name='values_layer1')(embedding_layer)
attention_kernel1 = layers.Attention()([query1, values1])

query2 = layers.Dense(units, name='query_layer2')(embedding_layer)
values2 = layers.Dense(units, name='values_layer2')(embedding_layer)
attention_kernel2 = layers.Attention()([query2, values2])
'''
concated_heads = layers.Concatenate()([attention_kernel0, attention_kernel1, attention_kernel2])
attention_output = layers.Dense(length_of_one_rna)(concated_heads)

RNNS_input = layers.Reshape((3000, -1))(attention_output)
# RNNS_input = layers.Reshape((3000, -1))(attention_kernel0)
# RNNS_input = layers.Flatten()(RNNS_input)

# Add & Norm
# Add = tf.keras.layers.Add()([RNNS_input, padded_input])

# feed_forward
hidden = layers.Dense(5, activation='relu')(RNNS_input)
hidden = layers.Dense(5, activation='relu')(hidden)

# Conv1d = layers.Conv1D(32, 1)(RNNS_input)
# Conv1d = layers.Conv1D(64, 3)(Conv1d)
# Conv1d = layers.Conv1D(128, 3)(Conv1d)
'''

GRU_layer = layers.GRU(100, activation='tanh', recurrent_activation='sigmoid', dropout=0.5)(embedding_layer)
#hidden = layers.Dense(100)(GRU_layer)
hidden = layers.Dense(100, activation='relu')(GRU_layer)
cf = layers.Dense(1, activation='sigmoid')(hidden)
# cf = layers.Dense(1, activation='sigmoid')(hidden)
# cf = layers.Dense(1)(GRU_layer)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.0004,
    beta_1=0.0,
    beta_2=0.0,
    epsilon=1e-07,
    amsgrad=False)
sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.8, nesterov=True)

classifier = models.Model(input_layer1, cf)

classifier.compile(optimizer=adam,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

classifier.summary()
plot_model(classifier, to_file="model.png")
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

#tf.keras.callbacks.TensorBoard(
    #log_dir='logs', histogram_freq=0, write_graph=True,
    #write_images=False, write_steps_per_second=False, update_freq='epoch',
    #profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
history = classifier.fit(padded_inputs, train_labels, batch_size=1, epochs=10, validation_data=(padded_tests, test_labels))
#history.history

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = classifier.evaluate(padded_tests, test_labels)
print("test loss, test acc:", results)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

classifier.save('./epoch_10_attention.h5')
# classifier.save_weights('./epoch_10_attention.h5')
# my_model = load_model('./epoch_10_attention.h5')
# my_model.get_weights()

