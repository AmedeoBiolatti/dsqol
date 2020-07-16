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
        metrics: typing.List[keras.metrics.Metric] = None,
        validation_data=None
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
    def val_step(data):
        x, y = data
        for m in metrics:
            m.update_state(y, model(x))
        pass

    @tf.function
    def val_step_distributed(data):
        strategy.run(val_step, args=(data,))
        pass

    # distributed dataset
    dist_data = strategy.experimental_distribute_dataset(data)
    train_data_iter = iter(dist_data)
    if validation_data is not None and metrics is not None:
        dist_validation_data = strategy.experimental_distribute_dataset(validation_data)
    else:
        dist_validation_data = None

    # custom training loop
    history = {'val_%s' % m.name: [] for m in metrics}
    for epoch in range(epochs):
        print("Epoch %3d:" % epoch, end="")
        t_start = time.time()
        step = 0
        for step in range(steps_per_epoch // steps_per_call):
            train_step(train_data_iter)
        t_end = time.time()
        dt = t_end - t_start
        print(" \tTime %3ds %3dms/step" % (int(dt), int(1000 * dt / (step + 1))), end="")
        #
        if dist_validation_data is not None:
            [m.reset_states() for m in metrics]
            for d in dist_validation_data:
                val_step_distributed(d)
            for m in metrics:
                res = m.result()
                history['val_%s' % m.name].append(res)
                print(" \tval_%s = %.3f" % (m.name, res), end="")
                pass
            pass
        #
        print()
    return history
