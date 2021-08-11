import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import MarginModels
import Evaluation
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.engine.base_layer import Layer
from Bio import SeqIO
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix



fasta_train = 'H_train.fasta'
csv_train = 'H_train.csv'
fasta_test = 'H_test.fasta'
csv_test = 'H_test.csv'


## Add label to the train dataset and generate X_train=record.seq and Y_train=label
size_train = 0
train_lst = []
train_samples = []
train_labels = []

letters = 'ACGT'
emb_dict = {letter: number + 1 for number, letter in
            enumerate(letters)}  # number+1 for emb bec

# test
with open(csv_train, 'w') as f:
    writer = csv.writer(f)
    for record in SeqIO.parse("H_test.fasta", "fasta"):
        label_train = 0 if 'CDS' in record.id else 1
        #print(label_train)
        tarin_sample = []
        writer.writerow([record.id, record.seq, len(record), label_train])
        size_train = size_train+1
        lst=[record.id, str(record.seq), len(record), label_train]
        #print(lst)
        for index, letter, in enumerate(record.seq):
            tarin_sample.append(emb_dict[letter])

        train_lst.append(lst)
        train_labels.append(label_train)
        train_samples.append(tarin_sample)


# padding
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(list(train_samples), maxlen=3000)

train_labels = to_categorical(train_labels, 2)
padded_inputs, train_labels = np.array(padded_inputs), np.array(train_labels)

max_train = 3000
length_of_one_rna = max_train


# test
size_test = 0
test_lst = []
test_samples = []
test_labels = []
with open(fasta_test)as fn:
    for record in SeqIO.parse("H_train.fasta", "fasta"):
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

# Model
input_layer1 = layers.Input(shape=(max_train,))
y_input_layer = layers.Input(shape=(2,))

embedding_layer = tf.keras.layers.Embedding(input_dim=5, output_dim=2, input_length=3000, mask_zero=True)(input_layer1)
flatt_output = layers.Flatten()(embedding_layer)

weight_decay = 1e-4
cf = MarginModels.ArcFace(2, regularizer=regularizers.l2(weight_decay))([flatt_output, y_input_layer])
#cf = SphereFace(2, regularizer=regularizers.l2(weight_decay))([attention_kernel0, y_input_layer])
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

classifier.fit([padded_inputs, train_labels], train_labels, epochs=5, batch_size=32, shuffle=True, validation_split=0.1)
classifier.save('./ArcFace.h5')
classifier.summary()
plot_model(classifier, to_file="model.png")

# prediction
pred = classifier.predict([padded_tests, test_labels])
print(pred)

pred = np.argmax(pred,axis=1)
print(pred)

# accuracy
CalculatedAccuracy = sum(pred == test_labels)/len(pred)
print(f'Accuracy: {CalculatedAccuracy:.3f}')

# evaluation scores
Evaluation.eval_metrics(test_labels, pred)

print(f'error: {mean_squared_error(test_labels, pred):.3f}')
print(f'accuracy: {accuracy_score(test_labels, pred):.3f}')
print(f'precision: {precision_score(test_labels, pred):.3f}')
print(f'Recall: {recall_score(test_labels, pred):.3f}')
print(f'F1_score: {f1_score(test_labels, pred):.3f}')

# confusion matrix
conf_matrix = confusion_matrix(test_labels, pred, normalize='true')
print(conf_matrix)

import seaborn as sns
sns.heatmap(conf_matrix, annot=True)

