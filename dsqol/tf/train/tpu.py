import tensorflow as tf
from tensorflow import keras
import typing
import time


def ftrain_distributed(
        model: keras.Model,
        data: tf.data.Dataset,
        strategy: tf.distribute.Strategy,
        epochs: int = 1,
        steps_per_epoch: int = None,
        steps_per_call: int = 1,
        loss: keras.losses.Loss = None,
        optimizer: keras.optimizers.Optimizer = None,
        metrics: typing.List[keras.metrics.Metric] = None
):
    if loss is None:
        loss = model.loss
    if optimizer is None:
        optimizer = model.optimizer
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

    # distributed dataset
    train_dist_ds = strategy.experimental_distribute_dataset(data)
    train_data_iter = iter(train_dist_ds)

    # custom training loop
    for epoch in range(epochs):
        print("Epoch %3d:" % epoch, end="")
        t_start = time.time()
        step = 0
        for step in range(steps_per_epoch // steps_per_call):
            train_step(train_data_iter)
        t_end = time.time()
        dt = t_end - t_start
        print(" \tTime %3ds %3dms/step" % (int(dt), int(1000 * dt / (step + 1))), end="")

        print()
    pass
