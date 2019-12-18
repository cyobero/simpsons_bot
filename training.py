"""Train Transformer model with customized loss function
"""
import tensorflow as tf
import os
from .model import Encoder, Decoder, BahdanauAttention

encoder = Encoder(vocab_inp_size,
                  embedding_dim,
                  units,
                  BATCH_SIZE)

attention_layer = BahdanauAttention(10)

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossEntropy(
    from_logits=True, reduction='None')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Set up checkpoints
CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



