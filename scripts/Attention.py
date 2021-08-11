import numpy as np
from Bio import SeqIO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.utils import plot_model
import keras
from matplotlib import pyplot as plt
from keras.models import load_model
from matplotlib import pyplot as plt
# import sklearn
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import dot_product_attention
import positional_embedding

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
d_model = 2
output_dim = 2

input_layer1 = layers.Input(shape=(max_train,))
embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2, input_length=3000)(input_layer1)
positional_embedding = layers.Lambda(positional_embedding.positional_encoding)([3000, 2])
add_embeddings = layers.Add()([embedding_layer, positional_embedding])
#flatt_output = layers.Flatten()(embedding_layer)

q0 = layers.Dense(d_model, use_bias=False, name='query_layer0')(embedding_layer)
k0 = layers.Dense(d_model, use_bias=False, name='key_layer0')(embedding_layer)
v0 = layers.Dense(d_model, use_bias=False, name='values_layer0')(embedding_layer)

attention_filter0 = tf.matmul(q0, k0, transpose_b=True)
scale = np.math.sqrt(d_model)
Scaling = attention_filter0 / scale
attention_weights0 = tf.nn.softmax(Scaling, axis=-1)
print('Attention weights are:')
print(attention_weights0.shape)
output0 = tf.matmul(attention_weights0, v0)
print('Output is:')
print(output0.shape)


q1 = layers.Dense(d_model, use_bias=False, name='query_layer1')(embedding_layer)
k1 = layers.Dense(d_model, use_bias=False, name='key_layer1')(embedding_layer)
v1 = layers.Dense(d_model, use_bias=False, name='values_layer1')(embedding_layer)

attention_filter1 = tf.matmul(q1, k1, transpose_b=True)
scale = np.math.sqrt(d_model)
Scaling = attention_filter1 / scale
attention_weights1 = tf.nn.softmax(Scaling, axis=-1)
print('Attention weights are:')
print(attention_weights1.shape)
output1 = tf.matmul(attention_weights1, v1)

concated_heads = layers.Concatenate()([output0, output1])
multi_head_output = layers.Dense(output_dim)(concated_heads)

#temp_out0, temp_attn0 = dot_product_attention.scaled_dot_product_attention(q0, k0, v0, None)

#q1 = layers.Dense(output_dim, use_bias=False, name='query_layer1')(embedding_layer)
#k1 = layers.Dense(output_dim, use_bias=False, name='key_layer1')(embedding_layer)
#v1 = layers.Dense(output_dim, use_bias=False, name='values_layer1')(embedding_layer)

#temp_out1, temp_attn1 = dot_product_attention.scaled_dot_product_attention(q1, k1, v1, None)

#q2 = layers.Dense(output_dim, use_bias=False, name='query_layer2')(embedding_layer)
#k2 = layers.Dense(output_dim, use_bias=False, name='key_layer2')(embedding_layer)
#v2 = layers.Dense(output_dim, use_bias=False, name='values_layer2')(embedding_layer)

#temp_out2, temp_attn2 = dot_product_attention.scaled_dot_product_attention(q2, k2, v2, None)
#concated_heads = layers.Concatenate()([temp_out0, temp_out1])
#attention_output = layers.Dense(output_dim)(temp_out0)

# feed_forward
#hidden = layers.Dense(3, activation='relu')(attention_output)
#hidden = layers.Dense(2, activation='relu')(hidden)


# cf = layers.Dense(1, activation='sigmoid')(temp_out)
cf = layers.Dense(1, activation='sigmoid')(multi_head_output)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.009,
    beta_1=0.6,
    beta_2=0.6,
    epsilon=1e-07,
    amsgrad=False)
#sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.8, nesterov=True)

classifier = models.Model(input_layer1, cf)

classifier.compile(optimizer=adam,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
classifier.summary()
#plot_model(classifier, to_file="model.png")
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

#tf.keras.callbacks.TensorBoard(
    #log_dir='logs', histogram_freq=0, write_graph=True,
    #write_images=False, write_steps_per_second=False, update_freq='epoch',
    #profile_batch=2, embeddings_freq=0, embeddings_metadata=None)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
history = classifier.fit(padded_inputs, train_labels, batch_size=32, epochs=5, validation_data=(padded_tests, test_labels))
'''
pred = classifier.predict([padded_tests, test_labels])
pred = np.argmax(pred, axis=1)

from sklearn.metrics import mean_squared_error, accuracy_score
print(f'error: {mean_squared_error(test_labels, pred)}')
print(f'accuracy: {accuracy_score(test_labels, pred)}')
#history.history

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = classifier.evaluate(padded_tests, test_labels)
print("test loss, test acc:", results)
'''
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
