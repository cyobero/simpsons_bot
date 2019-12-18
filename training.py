"""Train Transformer model with customized loss function.
First, create tf.data Dataset with hyperparams.
"""
import tensorflow as tf
import os
import time
import numpy as np
from model import *


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

EPOCHS = 20

@tf.function
def train_step(inp, output, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [output_line.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing -- feed target as next input
        for t in range(1, output.shape[1]):
            # pass enc_output to decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(output[:, t], predictions)

            # use teacher forcing
            dec_input = tf.expand_dims(output[:, t], 1)

        batch_loss = (loss / int(output.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply(zip(gradients, variables))

        return batch_loss


for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, output)) in enumerate(dataset.take(STEPS_PER_EPOCH)):
        batch_loss = train_step(inp, output, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(epoch+1,
                                                         batch,
                                                         batch_loss.numpy()))

    # save checkpoint every 3 epochs
    if (epoch + 1) % 3 == 0:
        checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

    print("Epoch {} Loss {:.4f}".format(epoch+1, total_loss/STEPS_PER_EPOCH))
    print("Time taken for 1 epoch: {}sec\n".format(time.time() - start))
