import tensorflow as tf
from tensorflow.keras import backend as K
import math


def _get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    rotation_matrix = get_3x3_mat([c1, s1, zero,
                                   -s1, c1, zero,
                                   zero, zero, one])
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)

    shear_matrix = get_3x3_mat([one, s2, zero,
                                zero, c2, zero,
                                zero, zero, one])
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one / height_zoom, zero, zero,
                               zero, one / width_zoom, zero,
                               zero, zero, one])
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one, zero, height_shift,
                                zero, one, width_shift,
                                zero, zero, one])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix, shift_matrix))


def Transform(
        dim=256,
        rot_mult=180.0,
        shear_mult=2.0,
        hzoom_mult=8.0,
        wzoom_mult=8.0,
        hshift_mult=8.0,
        wshift_mult=8.0
):
    def _transform(image):
        # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        # output - image randomly rotated, sheared, zoomed, and shifted
        XDIM = dim % 2  # fix for size 331

        rot = rot_mult * tf.random.normal([1], dtype='float32')
        shr = shear_mult * tf.random.normal([1], dtype='float32')
        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom_mult
        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom_mult
        h_shift = hshift_mult * tf.random.normal([1], dtype='float32')
        w_shift = wshift_mult * tf.random.normal([1], dtype='float32')

        # GET TRANSFORMATION MATRIX
        m = _get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

        # LIST DESTINATION PIXEL INDICES
        x = tf.repeat(tf.range(dim // 2, -dim // 2, -1), dim)
        y = tf.tile(tf.range(-dim // 2, dim // 2), [dim])
        z = tf.ones([dim * dim], dtype='int32')
        idx = tf.stack([x, y, z])

        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = K.cast(idx2, dtype='int32')
        idx2 = K.clip(idx2, -dim // 2 + XDIM + 1, dim // 2)

        # FIND ORIGIN PIXEL VALUES
        idx3 = tf.stack([dim // 2 - idx2[0, ], dim // 2 - 1 + idx2[1, ]])
        d = tf.gather_nd(image, tf.transpose(idx3))

        return tf.reshape(d, [dim, dim, 3])

    return _transform
