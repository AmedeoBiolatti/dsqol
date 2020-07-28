import typing
from tensorflow import keras


def average_weights(saves: typing.List[str], decay=1.0) -> keras.Model:
    total_decay = decay ** 0
    m0 = keras.models.load_model(saves[0])
    m1 = keras.models.load_model(saves[0])
    for i, save in enumerate(saves[1:]):
        m1.load_weights(save)
        total_decay += decay ** (i + 1)
        for w0, w1 in zip(m0.weights, m1.weights):
            w0.assign_add(decay ** (i + 1) * w1)
    del m1
    for w in m0.weights:
        w.assign(w / total_decay)
    return m0
