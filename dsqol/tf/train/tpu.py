import tensorflow as tf
from tensorflow import keras
import typing
import time
import numpy as np


def ftrain_distributed(
        model: keras.Model,
        data: tf.data.Dataset,
        strategy: tf.distribute.Strategy,
        epochs: int = 1,
        steps_per_epoch: int = None,
        steps_per_call: int = 1,
        loss: keras.losses.Loss = None,
        optimizer: keras.optimizers.Optimizer = None,
        metrics: typing.List[keras.metrics.Metric] = None,
        validation_data=None,
        validation_freq=1,
        val_steps_per_epoch=None,
        verbose=1
):
    if loss is None:
        loss = model.loss
    if optimizer is None:
        optimizer = model.optimizer
    if metrics is None:
        metrics = model.metrics
    if steps_per_epoch is None:
        steps_per_epoch = 100
        print("steps_per_epochs imposed to 100")

    @tf.function
    def train_step(data_iter):
        def train_step_fn(images, labels):
            with tf.GradientTape() as tape:
                probabilities = model(images, training=True)
                loss_value = loss(labels, probabilities)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # loop executing on TPU
        for _ in tf.range(steps_per_call):
            strategy.run(train_step_fn, next(data_iter))

    @tf.function
    def validation_step(data):
        xb, yb = data
        pred = model(xb)
        for m in metrics:
            m.update_state(yb, pred)
        pass

    # distributed dataset
    dist_data = strategy.experimental_distribute_dataset(data)
    train_data_iter = iter(dist_data)

    # custom training loop
    history = {'val_%s' % m.name: [] for m in metrics}
    for epoch in range(epochs):
        if verbose:
            print("Epoch %3d:" % epoch, end="")
        t_start = time.time()
        step = 0
        for step in range(steps_per_epoch // steps_per_call):
            train_step(train_data_iter)
        t_end = time.time()
        dt = t_end - t_start
        if verbose:
            print(" \tTime %3ds %3dms/step" % (int(dt), int(1000 * dt / (step + 1))), end="")
        #
        if (validation_data is not None) and len(metrics) > 0 and (step % validation_freq == 0) and (
                validation_freq == 1 or epoch > 0):
            [m.reset_states() for m in metrics]
            for step, data_batch in enumerate(validation_data):
                validation_step(data_batch)
                pass
            for m in metrics:
                res = m.result()
                history['val_%s' % m.name].append(res)
                if verbose:
                    print(" \tval_%s = %.3f" % (m.name, res), end="")
                pass
            pass
        #
        print()
    return history


def fpredict_distributed(model: keras.Model, ds: tf.data.Dataset, strategy: tf.distribute.Strategy, steps=None,
                         verbose=0):
    dist_data = strategy.experimental_distribute_dataset(ds)
    dist_data_iter = iter(dist_data)
    steps = int(1e3) if steps is None else steps

    @tf.function
    def predict_step(data_iter):
        batch = next(data_iter)
        if isinstance(batch, tuple) and len(batch) > 1:
            batch = batch[0]
        pred = model(batch)
        return pred

    @tf.function
    def predict_step_distributed(data_iter):
        preds = strategy.run(predict_step, args=(data_iter,))
        return preds

    preds = []
    try:
        step = 0
        for step in range(steps):
            preds.append(predict_step_distributed(dist_data_iter).numpy())
            pass
    except tf.errors.OutOfRangeError:
        if verbose:
            print("Iterating ended at step %d" % step)
        pass
    preds = np.concatenate(preds, axis=0)
    return preds
