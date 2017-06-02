import argparse
import cPickle as pickle
import csv
import sys
from keras.utils import generic_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from itertools import islice
from random import shuffle
import os

import embedding as emb

from model import *

MAX_NB_WORDS=1000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100

M = Model()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model/test")
    parser.add_argument("--data_path", default="data/studystack_qa_cleaner_no_qm.txt")
    parser.add_argument("--csv_file", default="data/train.tsv")
    parser.add_argument("--load_tokenizer", default="model/tokenizer_studystack_full.pkl")
    parser.add_argument("--macrobatch_size", type=int, default=1000)
    parser.add_argument("--min_margin", type=float, default=0)
    parser.add_argument("--max_margin", type=float, default=0.2)
    parser.add_argument("--load_model")
    parser.add_argument("--load_arch")
    parser.add_argument("--save_arch")
    add_model_params(parser)
    add_training_params(parser)
    add_data_params(parser)
    args = parser.parse_args()

    print "Loading tokenizer..."
    tokenizer = load_tokenizer(args.load_tokenizer)
    vocab_size = vocabulary_size(tokenizer)
    print "Vocabulary size:", vocab_size

    print "Loading GLOVE.G tokenizer..."
    embeddings_index = {}
    f = open(os.path.join("data/glove.6B", 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    print "Creating model..."
    model = M.create_model(args)

    print "Summary of model..."
    model.summary()

    print "Compiling model..."
    M.compile_model(model, args)

    print "Loading training data..."
    generator = generate_training_data(args.data_path, embeddings_index=embeddings_index, model=model, args=args)

    print "Fitting model..."
    for epoch in xrange(args.epochs):
        progbar = generic_utils.Progbar(args.samples_per_epoch)
        n = 0
        while n < args.samples_per_epoch:
            X, y = next(generator)
            print "X.shape:", X.shape
            loss = model.train_on_batch(X, y)
            bs = X.shape[0]
            progbar.add(bs, values=[('train loss', loss)])
            n += bs

## embedding
def text_to_data(lines, embeddings_index):
    # sequences = tokenizer.texts_to_sequences(lines)
    # # apply maxlen limitation only when sequences are longer
    # seqmaxlen = max([len(s) for s in sequences])
    # if seqmaxlen > maxlen:
    #     seqmaxlen = maxlen
    # data = pad_sequences(sequences, maxlen=seqmaxlen)
    # return data
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    word_index = tokenizer.word_index
    print('Found %s unique tokens. And num of lines is %s ' % (len(word_index), len(lines)))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = np.array(embedding_vector)
        else:
            embedding_matrix[i] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    embedding = []
    for row in data:
        rr = [embedding_matrix[i] for i in row]
        embedding.append(rr)

    return np.asarray(embedding)

def predict_data(model, data, args):
    if args.bidirectional:
        pred = model.predict({'input': data}, batch_size=args.batch_size, verbose=args.verbose)
        pred = pred['output']
    else:
        pred = model.predict(data, batch_size=args.batch_size, verbose=args.verbose)

    return pred

def generate_training_data(data_path, embeddings_index, model, args):
    while True:
        print 'opening data_path %s' % data_path
        with open(data_path) as f:
            csv.field_size_limit(sys.maxsize)
            reader = csv.reader(f, delimiter="\t", strict=True, quoting=csv.QUOTE_NONE)

            while True:
            # read macrobatch_size lines from reader
                lines = list(islice(reader, args.macrobatch_size))
                #print "Lines:", len(lines)
                if not lines:
                    break;


                print "Sample lines:"
                print lines[0]
                print lines[1]
                print lines[2]

                print zip(*lines[:3])

                shuffle(lines)
                ids, questions, answers = zip(*lines)
                print "ids:", len(ids), "questions:", len(questions), "answers:", len(answers)

                texts = questions + answers
                print "texts:", len(texts)
                embedding = text_to_data(texts, embeddings_index)
                print "data:", embedding.shape

                pred = predict_data(model, embedding, args)
                print "pred:", pred.shape
                half = int(pred.shape[0] / 2)
                question_vectors = pred[0:half]
                answer_vectors = pred[half:]
                print "question_vectors:", question_vectors.shape, "answer_vectors.shape", answer_vectors.shape
                dists = pairwise_distances(question_vectors, answer_vectors, metric="cosine", n_jobs=1)
                print "distances:", dists.shape

                X = np.empty((args.batch_size, embedding.shape[1], embedding.shape[2]))
                y = np.empty((args.batch_size, args.hidden_size))
                n = 0
                produced = 0
                total_pa_dist = 0
                total_na_dist = 0
                total_margin = 0
                for i in xrange(len(questions)):
                    sorted = np.argsort(dists[i])
                    #print ""
                    #print "question %d:" % i, questions[i]
                    for j in sorted:
                        margin = dists[i,j] - dists[i,i] # ideally margin should be negative
                        # print "answer %d:" % j, answers[j], "(correct answer: %s)" % answers[i]
                        # print "distance:", dists[i,j], dists[i, i], "(margin %f)" % margin
                        if j != i and answers[j].strip().lower() != answers[i].strip().lower() \
                                and (args.min_margin is None or margin > args.min_margin):
                            if (args.max_margin is None or margin < args.max_margin):
                                X[n] = embedding[i]
                                X[n+1] = embedding[half + i]
                                X[n+2] = embedding[half + j]
                                n += 3
                                print "Question:", questions[i], texts[i]
                                print "Right answer:", answers[i], texts[half + i]
                                print "Wrong answer:", answers[j], texts[half + j]
                                if n == args.batch_size:
                                    yield X, y
                                    n = 0

                                total_pa_dist += dists[i,i]
                                total_na_dist += dists[i,j]
                                total_margin += margin
                                produced += 1
                            break

                    if n > 0:
                        yield X[:n], y[:n]

                    print ""
                    print "Read %d lines, used %d questions, discarded %d" % (len(lines), produced, len(lines) - produced)
                    print "Average right answer distance %g, wrong answer distance %g, margin %g" % \
                        (total_pa_dist / produced, total_na_dist / produced, total_margin / produced)
                    print ""


def add_model_params(parser):
    parser.add_argument("--rnn", choices=["LSTM", "GRU"], default="GRU")
    parser.add_argument("--embed_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bidirectional", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--dense_layers", type=int, default=0)
    parser.add_argument("--dense_activation", choices=['relu','sigmoid','tanh'], default='relu')
    parser.add_argument("--convolution", action='store_true', default=False)
    parser.add_argument("--conv_filters", type=int, default=1000)
    parser.add_argument("--conv_filter_length", type=int, default=3)
    parser.add_argument("--conv_activation", choices=['relu','sigmoid','tanh'], default='relu')
    parser.add_argument("--conv_subsample_length", type=int, default=1)
    parser.add_argument("--conv_border_mode", choices=['valid','same'], default='valid')
    parser.add_argument("--pooling", action='store_true', default=False)
    parser.add_argument("--pool_length", type=int, default=2)
    parser.add_argument("--loss", choices=['cosine', 'gesd', 'aesd'], default='cosine')


def add_training_params(parser):
    parser.add_argument("--validation_split", type=float, default=0)
    parser.add_argument("--optimizer", choices=['adam', 'rmsprop', 'sgd'], default='adam')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_epochs", type=int, default=0)
    parser.add_argument("--samples_per_epoch", type=int, default=1500000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)


def add_data_params(parser):
    parser.add_argument("--max_words", type=int)
    parser.add_argument("--maxlen", type=int, default=255)


def load_tokenizer(tokenizer_path):
    return pickle.load(open(tokenizer_path, "rb"))


def vocabulary_size(tokenizer):
    return tokenizer.nb_words+1 if tokenizer.nb_words else len(tokenizer.word_index)+1

if __name__ == '__main__':
    main()
