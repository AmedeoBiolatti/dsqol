import tensorflow as tf


def Mixup(dim, batch_size, probability=1.0):
    def _mixup(img, lab):
        DIM = dim
        imgs = []
        labs = []
        for j in range(batch_size):
            do = tf.cast(tf.random.uniform([], 0, 1) <= probability, tf.float32)
            alpha = (1 - do) * tf.cast(tf.random.uniform([], 0, 1), tf.float32)
            k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
            new_img = alpha * img[j] + (1 - alpha) * img[k]
            new_lab = alpha * lab[j] + (1 - alpha) * lab[k]
            imgs.append(new_img)
            labs.append(new_lab)
        img2 = tf.reshape(tf.stack(imgs), (batch_size, DIM, DIM, 3))
        lab2 = tf.reshape(tf.stack(labs), (batch_size, 1))
        return img2, lab2

    return _mixup
