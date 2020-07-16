import tensorflow as tf


def Cutmix(dim, batch_size, probability=1.0):
    def _cutmix(image, label):
        # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
        # output - a batch of images with cutmix applied
        imgs = []
        labs = []
        for j in range(batch_size):
            # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
            P = tf.cast(tf.random.uniform([], 0, 1) <= probability, tf.int32)
            # CHOOSE RANDOM IMAGE TO CUTMIX WITH
            k = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
            # CHOOSE RANDOM LOCATION
            x = tf.cast(tf.random.uniform([], 0, dim), tf.int32)
            y = tf.cast(tf.random.uniform([], 0, dim), tf.int32)
            b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
            WIDTH = tf.cast(dim * tf.math.sqrt(1 - b), tf.int32) * P
            ya = tf.math.maximum(0, y - WIDTH // 2)
            yb = tf.math.minimum(dim, y + WIDTH // 2)
            xa = tf.math.maximum(0, x - WIDTH // 2)
            xb = tf.math.minimum(dim, x + WIDTH // 2)
            # MAKE CUTMIX IMAGE
            one = image[j, ya:yb, 0:xa, :]
            two = image[k, ya:yb, xa:xb, :]
            three = image[j, ya:yb, xb:dim, :]
            middle = tf.concat([one, two, three], axis=1)
            img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:dim, :, :]], axis=0)
            imgs.append(img)
            # MAKE CUTMIX LABEL
            a = tf.cast(WIDTH * WIDTH / dim / dim, tf.float32)
            lab1 = tf.cast(label[j,], tf.float32)
            lab2 = tf.cast(label[k,], tf.float32)
            labs.append((1 - a) * lab1 + a * lab2)

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
        image2 = tf.reshape(tf.stack(imgs), (batch_size, dim, dim, 3))
        label2 = tf.reshape(tf.stack(labs), (batch_size, 1))
        return image2, label2

    return _cutmix
