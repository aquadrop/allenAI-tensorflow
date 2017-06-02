from keras.layers import recurrent
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Activation, Dropout
import tensorflow as tf

class Model:

    MAX_NB_WORDS = 1000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 100

    word_index = None
    embedding_matrix = None

    def __init__(self):
        return


    def create_model(self, args):
        assert args.batch_size % 3 == 0, "Batch size must be multiple of 3"

        if args.rnn == 'GRU':
            RNN = recurrent.GRU
        elif args.rnn == 'LSTM':
            RNN = recurrent.LSTM
        else:
            assert False, "Invalid RNN"

  #   if args.bidirectional:
  #       assert not args.convolution, "Convolutional layer is not supported with bidirectional RNN"
  #       assert not args.pooling, "Pooling layer is not supported with bidirectional RNN"
  #       assert args.dense_layers == 0, "Dense layers are not supported with bidirectional RNN"
  #       model = Sequential()
  #       model.add_input(name="input", batch_input_shape=(args.batch_size,1), dtype="uint")
  #       model.add_node(Embedding(vocab_size, args.embed_size, mask_zero=True), name="embed", input='input')
  #       for i in xrange(args.layers):
  #           model.add_node(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True),
  #               name='forward'+str(i+1),
  #               input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None,
  #               inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
  #           model.add_node(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True, go_backwards=True),
  #               name='backward'+str(i+1),
  #               input='embed' if i == 0 else 'dropout'+str(i) if args.dropout > 0 else None,
  #               inputs=['forward'+str(i), 'backward'+str(i)] if i > 0 and args.dropout == 0 else [])
  #       if args.dropout > 0:
  #           model.add_node(Dropout(args.dropout), name='dropout'+str(i+1), inputs=['forward'+str(i+1), 'backward'+str(i+1)])
  #   model.add_output(name='output',
  #       input='dropout'+str(args.layers) if args.dropout > 0 else None,
  #       inputs=['forward'+str(args.layers), 'backward'+str(args.layers)] if args.dropout == 0 else [])
  #   assert args.dense_layers == 0, "Bidirectional model doesn't support dense layers yet"
  # else:
        print 'using conv model...'
        model = Sequential()
        print 'adding input layer'
        # embedding_layer = Embedding(len(self.word_index) + 1,
        #                             self.EMBEDDING_DIM,
        #                             weights=[self.embedding_matrix],
        #                             input_length=self.MAX_SEQUENCE_LENGTH,
        #                             trainable=False)
        # model.add(Embedding(vocab_size, args.embed_size, mask_zero=not args.convolution))
        model.add(RNN(args.hidden_size, input_dim=self.EMBEDDING_DIM, input_length=self.MAX_SEQUENCE_LENGTH, return_sequences=True))
        if args.convolution:
          print '1d conv layer...'
          model.add(Convolution1D(nb_filter=args.conv_filters,
                              filter_length=args.conv_filter_length,
                              border_mode=args.conv_border_mode,
                              activation=args.conv_activation,
                              subsample_length=args.conv_subsample_length))
          if args.pooling:
            model.add(MaxPooling1D(pool_length=args.pool_length))
        for i in xrange(args.layers):
          print 'stacking layer', str(i), args.hidden_size
          model.add(RNN(args.hidden_size, return_sequences=False if i + 1 == args.layers else True))
          if args.dropout > 0:
            model.add(Dropout(args.dropout))
        for i in xrange(args.dense_layers):
          if i + 1 == args.dense_layers:
            model.add(Dense(args.hidden_size, activation='linear'))
          else:
            model.add(Dense(args.hidden_size, activation=args.dense_activation))

        return model


    def compile_model(self, model, args):
        def mean(x, axis=None, keepdims=False):
            return tf.reduce_mean(x, axis=axis)

        def l2_normalize(x, axis):
            norm = tf.reshape(1 / tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis)), shape=[-1, 1])
            return x * norm

        def cosine_similarity(y_true, y_pred):
            # assert y_true.get_shape() == 2
            # assert y_pred.get_shape() == 2
            y_true = l2_normalize(y_true, axis=1)
            y_pred = l2_normalize(y_pred, axis=1)
            return tf.reduce_sum(y_true * y_pred, axis=1)

        def cosine_ranking_loss(y_true, y_pred):
            q = y_pred[0::3]
            a_correct = y_pred[1::3]
            a_incorrect = y_pred[2::3]

            positive = tf.maximum(0., args.margin - cosine_similarity(q, a_correct) + cosine_similarity(q, a_incorrect))
            return tf.reduce_mean(positive, axis=-1)

        loss = cosine_ranking_loss

        if args.bidirectional:
            model.compile(optimizer=args.optimizer, loss={'output': loss})
        else:
            model.compile(optimizer=args.optimizer, loss=loss)

## end