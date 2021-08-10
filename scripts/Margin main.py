import numpy as np
import pandas as pd
import tensorflow as tf
import MarginModels  # scripts
import Evaluation # scripts
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import activations
from tensorflow.python.keras.engine.base_layer import Layer
from Bio import SeqIO
# import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns


fasta_train = 'H_train.fasta'
csv_train = 'H_train.csv'
fasta_test = 'H_test.fasta'
csv_test = 'H_test.csv'


# Add label to the train dataset and generate X_train=record.seq and Y_train=label
size_train = 0
train_lst = []
train_samples = []
train_labels = []

letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb bec


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
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(list(train_samples), maxlen=3000)

max_train = 3000
length_of_one_rna = max_train

train_labels = to_categorical(train_labels, 2)
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

# padding test
padded_tests = tf.keras.preprocessing.sequence.pad_sequences(list(test_samples), maxlen=3000)
padded_tests, test_labels = np.array(padded_tests), np.array(test_labels)


# Model
input_layer1 = layers.Input(shape=(max_train,))
y_input_layer = layers.Input(shape=(2,))


embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2, input_length=3000, mask_zero=True)(input_layer1)
flatt_output = layers.Flatten()(embedding_layer)

##Regularizers allow you to apply penalties on layer parameters.
##These penalties are summed into the loss function that the network optimizes.
weight_decay = 1e-4
cf = MarginModels.ArcFace(2, regularizer=regularizers.l2(weight_decay))([flatt_output, y_input_layer])
#cf = SphereFace(2, regularizer=regularizers.l2(weight_decay))([flatt_output, y_input_layer])
#cf =CosFace(2, regularizer=regularizers.l2(weight_decay))([flatt_output, y_input_layer])


adam = tf.keras.optimizers.Adam(
    learning_rate=0.0009,
    beta_1=0.6,
    beta_2=0.6,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam")

classifier = models.Model([input_layer1, y_input_layer] , cf)

classifier.compile(loss='categorical_crossentropy',
                   optimizer=adam,
                   metrics=['accuracy'])

# fit the model
classifier.fit([padded_inputs, train_labels], train_labels, epochs=10, batch_size=32, shuffle=True, validation_split=0.1)

classifier.summary()
tf.keras.utils.plot_model(classifier, to_file="model.png")

# prediction
pred = classifier.predict([padded_tests, test_labels])
print(pred)

pred = np.argmax(pred,axis=1)
print(pred)

# acc
CalculatedAccuracy = sum(pred == test_labels)/len(pred)
print(f'Accuracy: {CalculatedAccuracy:.3f}')


# all eval_metrics
print(f'error: {mean_squared_error(test_labels, pred):.3f}')
# Accuracy
print(f'accuracy: {accuracy_score(test_labels, pred):.3f}')
# Precision
print(f'precision: {precision_score(test_labels, pred):.3f}')
# Recall
print(f'Recall: {recall_score(test_labels, pred):.3f}')
# F1 score
print(f'F1_score: {f1_score(test_labels, pred):.3f}')

# confusion Matrix
print('Confusion Matrix')

conf_matrix = confusion_matrix(test_labels, pred, normalize='true')
print(conf_matrix)

sns.heatmap(conf_matrix, annot=True)

Evaluation.eval_metrics(test_labels, pred)

