"""Build seq2seq model, which consists of:
    1. Encoder
    2. Decoder
    3. Attention
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocess import max_length, load_dataset, FILE_PATH

class Encoder(tf.keras.Model):
    """Encoder Layer
    Encoder layer takes an input of words and outputs word vectors to be
    used as inputs to Decoder layer.
    """
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')


    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


    def initialize_hidden_state(self):
        """Initializes hidden state"""
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """Attention layer using Bahdanau Attention"""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def call(self, query, values):
        """
        hidden_shape == (batch_size, hidden_size)
        hidden_shape_with_time_axis == (batch_size, 1, hidden_size)
        We do this to perform addition to calculate the score
        """
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_state)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):


    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)


    def call(self, x, hidden, enc_output):
        # enc_output == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # pass concatenated vector to GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


input_tensor, output_tensor, input_line, output_line = load_dataset(FILE_PATH)

# Calculate max length of target tensors
max_length_output, max_length_input = max_length(output_tensor), max_length(input_tensor)

# Create trainining/test set with 80-20 split
input_tensor_train, input_tensor_test, output_tensor_train, output_tensor_test = train_test_split(input_tensor, output_tensor, test_size=0.2)

# Create tf.data Dataset. Start by creating hyperparameters.
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
STEPS_PER_EPOCH = len(input_tensor_train) // BATCH_SIZE
EMBEDDING_DIM = 256
UNITS = 1024
VOCAB_INPUT_SIZE = len(input_line.word_index) + 1
VOCAB_OUTPUT_SIZE = len(output_line.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, output_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(VOCAB_INPUT_SIZE,
                  EMBEDDING_DIM,
                  UNITS,
                  BATCH_SIZE)

decoder = Decoder(VOCAB_OUTPUT_SIZE,
                  EMBEDDING_DIM,
                  UNITS,
                  BATCH_SIZE)

attention_layer = BahdanauAttention(10)
