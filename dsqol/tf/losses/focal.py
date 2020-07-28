import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=None, from_logits=False,
                 reduction=keras.losses.Reduction.NONE, name=None):
        super(BinaryFocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            pred_prob = tf.sigmoid(y_pred)
        else:
            pred_prob = y_pred
        y_true = self.label_smoothing + (1 - 2 * self.label_smoothing) * y_true
        ce = keras.losses.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits,
                                              label_smoothing=self.label_smoothing)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)

        return alpha_factor * modulating_factor * ce
