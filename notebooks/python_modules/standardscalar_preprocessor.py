# from .preprocessor import Preprocessor
#
#
# class StandardScalarPreprocessor(Preprocessor):
#     def __init__(self):
#         super(StandardScalarPreprocessor, self).__init__()
#
from typing import Dict, Text, Any
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.python.framework import sparse_tensor

from google.protobuf import json_format
from tensorflow.python.lib.io import file_io
import json

# from tensorflow_transform import analyzers

NUMERICAL_FEATURE_KEYS = set()
CATEGORICAL_FEATURE_KEYS = set()
LABEL_COLUMN = 'isfraud'
VOCAB_SIZE = 1000
OOV_SIZE = 10

# TRAIN_BATCH_SIZE = 100
# EVAL_BATCH_SIZE = 500
# from anomaly_detection_tfx.utils import base_utils


def _transform_key(x: Text) -> Text:
    return f"{x}_transform"


def _identify_columns(key, tensor):
    if key == LABEL_COLUMN:
        return
#     print(f"key = {key} tensor = {tensor.dtype}")
    if tensor.dtype == tf.dtypes.string:
        CATEGORICAL_FEATURE_KEYS.add(key)
    else:
        NUMERICAL_FEATURE_KEYS.add(key)


def _preprocess_categorical_features(categorical_feature_keys, outputs):
    for key in categorical_feature_keys:
        input_tensor = outputs[key]
        if isinstance(input_tensor, tf.sparse.SparseTensor):
            dense = tf.sparse.to_dense(tf.SparseTensor(input_tensor.indices, input_tensor.values,[input_tensor.dense_shape[0], 1]), default_value='', name=f"{key}_to_dense")
        else:
            dense = input_tensor
        outputs[key] = tft.compute_and_apply_vocabulary(
            dense,
            num_oov_buckets=OOV_SIZE,
            vocab_filename=f"{key}_vocab")

#         outputs[key] = dense
    

def _preprocess_numerical_features(numerical_feature_keys, outputs):
    for key in numerical_feature_keys:
        input_tensor = outputs[key]
        if isinstance(input_tensor, tf.sparse.SparseTensor):
            if "int" in input_tensor.dtype.name:
                dense_temp = tf.sparse.to_dense(tf.SparseTensor(input_tensor.indices, input_tensor.values,[input_tensor.dense_shape[0], 1]), default_value=0, name=f"{key}_to_dense")
                dense = dense_temp
            elif "float" in input_tensor.dtype.name:
                dense_temp = tf.sparse.to_dense(tf.SparseTensor(input_tensor.indices, input_tensor.values,[input_tensor.dense_shape[0], 1]), default_value=0.0, name=f"{key}_to_dense")
                dense = dense_temp
            outputs[key] = tft.scale_to_0_1(dense)
        else:
            outputs[key] = tft.scale_to_0_1(outputs[key])

def _preprocess_label_column(outputs):
    input_tensor = outputs[LABEL_COLUMN]
    dense_temp = tf.sparse.to_dense(tf.SparseTensor(input_tensor.indices, input_tensor.values,[input_tensor.dense_shape[0], 1]), default_value=0, name=f"{LABEL_COLUMN}_to_dense")
    outputs[LABEL_COLUMN] = dense_temp

def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    outputs = inputs.copy()

    for key in outputs.keys():
        _identify_columns(key, outputs[key])
    _preprocess_categorical_features(CATEGORICAL_FEATURE_KEYS, outputs)
    _preprocess_numerical_features(NUMERICAL_FEATURE_KEYS, outputs)
    _preprocess_label_column(outputs)
    
#     print("***********************************************************")
#     for key in outputs.keys():
#         print(f"key => {key} \noutput = {outputs[key]} \ninput = {inputs[key].shape} \n\n\n")
#     print("***********************************************************")
    return outputs
