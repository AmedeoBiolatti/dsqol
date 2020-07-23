import tensorflow as tf
import typing
import numpy as np


def separate_by_target(ds: tf.data.Dataset, idx: int = 1, thr: float = 0.5
                       ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    def _cond0(*args):
        return tf.cast(args[idx], tf.float32) < thr

    def _cond1(*args):
        return tf.cast(args[idx], tf.float32) >= thr

    ds0 = ds.filter(_cond0)
    ds1 = ds.filter(_cond1)

    return ds0, ds1


def merge_ds(ds0: tf.data.Dataset, ds1: tf.data.Dataset, pos_ratio: float = 0.5) -> tf.data.Dataset:
    n1 = int(1000 * pos_ratio)
    n0 = 1000 - n1
    choice_ds = tf.data.Dataset.from_tensor_slices(
        np.concatenate([np.zeros(n0), np.ones(n1)]).astype('int64')).shuffle(n0 + n1).repeat()
    ds = tf.data.experimental.choose_from_datasets([ds0, ds1],
                                                   choice_ds.prefetch(tf.data.experimental.AUTOTUNE))
    return ds


def balance_ds(ds: tf.data.Dataset, pos_ratio: float = 0.5, idx: int = 1, thr: float = 0.5):
    ds0, ds1 = separate_by_target(ds, idx=idx, thr=thr)
    return merge_ds(ds0, ds1, pos_ratio=pos_ratio)
