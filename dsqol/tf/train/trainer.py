import tensorflow as tf
from tensorflow import keras
import re, typing, time
import pandas as pd
import matplotlib.pyplot  as plt


class SessionData:
    epochs: typing.Dict[str, typing.List[int]] = dict()
    metrics: typing.Dict[str, typing.List[float]] = dict()
    colors: typing.Dict[str, str] = dict()
    styles: typing.Dict[str, str] = dict()
    alphas: typing.Dict[str, float] = dict()

    _train_val_use_same_color = True
    total_time: float = -1.0

    def __init__(self):
        self._tic()

    def _tic(self):
        self.total_time = time.time()

    def _toc(self):
        self.total_time = (time.time() - self.total_time)

    def add(self, name, epoch, value):
        if name not in self.epochs.keys():
            self.epochs[name] = []
            self.metrics[name] = []
            pass
        self.epochs[name].append(epoch)
        self.metrics[name].append(value)
        pass

    def get_as_df(self, pattern=None) -> pd.DataFrame:
        keys = list(self.epochs.keys())
        if pattern is not None:
            keys = [k for k in keys if re.match(pattern, k)]
        df = pd.DataFrame({'epoch': []})
        for k in keys:
            df = pd.merge(df, pd.DataFrame({'epoch': self.epochs[k], k: self.metrics[k]}), on='epoch', how='outer')
        df = df.sort_values('epoch').set_index('epoch')
        return df

    def plot(self, pattern=None, **kwargs):
        df = self.get_as_df(pattern=pattern)
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)
        used_colors = dict()
        for col in df.columns:
            color = self.colors[col] if col in self.colors.keys() else None
            if color is None and self._train_val_use_same_color:
                if re.match("^val_", col):
                    if re.sub("^val_", "", col) in used_colors.keys():
                        color = used_colors[re.sub("^val_", "", col)]
                else:
                    if ("val_%s" % col) in used_colors.keys():
                        color = used_colors["val_%s" % col]
            #
            style = self.styles[col] if col in self.styles.keys() else None
            alpha = self.alphas[col] if col in self.alphas.keys() else None
            line = ax.plot(df.index, df[col], color=color, linestyle=style, alpha=alpha, label=col)
            plt.xlabel('Epoch')
            plt.xticks(df.index)
            used_colors[col] = line[0].get_color()
            pass
        plt.legend()
        pass

    pass


class Trainer:
    verbose: int = 0

    def __init__(self, verbose=0):
        self.verbose = verbose
        pass

    def _log(self, message, level=1):
        if self.verbose >= level:
            print(message)

    def __call__(self,
                 model: keras.Model,
                 ds: tf.data.Dataset,
                 epochs=None,
                 steps_per_epoch=None,
                 validation_data: tf.data.Dataset = None,
                 validation_freq=1) -> SessionData:
        out = self.call(model, ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_data,
                        validation_freq=validation_freq)
        out._toc()
        return out

    def call(self, model: keras.Model, ds: tf.data.Dataset, epochs=None, steps_per_epoch=None,
             validation_data: tf.data.Dataset = None,
             validation_freq=None) -> SessionData:
        raise NotImplementedError

    pass


class BaseTrainer(Trainer):
    def call(self, model: keras.Model, ds: tf.data.Dataset, epochs=None, steps_per_epoch=None,
             validation_data: tf.data.Dataset = None,
             validation_freq=1) -> SessionData:
        sd = SessionData()
        h = model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_data,
                      validation_freq=validation_freq, verbose=self.verbose)
        for k in h.history.keys():
            val = re.match("^val_", k)
            if val:
                sd.styles[k] = '--'
            for e, v in enumerate(h.history[k]):
                # TODO what if validation_freq > 1?
                sd.add(k, e * (validation_freq if val else 1), v)
            pass
        return sd


class CustomTrainer(Trainer):
    def call(self, model: keras.Model, ds: tf.data.Dataset, epochs=None, steps_per_epoch=None,
             validation_data: tf.data.Dataset = None,
             validation_freq=1) -> SessionData:
        sd = SessionData()

        loss: keras.losses.Loss = model.loss
        optimizer: keras.optimizers.Optimizer = model.optimizer

        @tf.function
        def epoch_fn(data_iter):
            for step in range(steps_per_epoch):
                x, y = next(data_iter)
                with tf.GradientTape() as tape:
                    p = model(x, training=True)
                    loss_value = loss(y, p)
                    pass
                grad = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))
            pass

        ds_iter = iter(ds)
        for epoch in range(epochs):
            self._log("Epoch %3d/%3d" % (epoch, epochs))
            epoch_fn(ds_iter)
            pass

        return sd
