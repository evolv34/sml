from typing import List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

# from anomaly_detection_tfx.constants import Metrics
# from anomaly_detection_tfx.models import FraudDetectDenseModel
# from anomaly_detection_tfx.utils import base_utils
from tensorflow.keras import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model
import math
import json
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2

NUMERICAL_FEATURE_KEYS = set()
CATEGORICAL_FEATURE_KEYS = set()
LABEL_COLUMN = 'isfraud'
VOCAB_SIZE = 1000
OOV_SIZE = 10

TRAIN_BATCH_SIZE = 100
EVAL_BATCH_SIZE = 500


class FraudDetectDenseModel(Model):

    def __init__(self, feature_columns, num_classes=2):
        super(FraudDetectDenseModel, self).__init__()
        self.sequential = keras.Sequential()

        inputs = [
            self.sequential.add(layers.Input(shape=(1,), name=f))
            for f in feature_columns
        ]

        self.sequential.add(layers.Dense(128, activation='relu', name="dense_layer_1"))
        self.sequential.add(layers.Dense(128, activation='relu', name="dense_later_2"))
        self.sequential.add(layers.Dropout(.1, name="dropout"))
        self.sequential.add(layers.Dense(math.ceil(num_classes / 2), activation='sigmoid', name="output"))

    def call(self, inputs, training=False, mask=None):
        output = self.sequential(inputs)
        return output
    
def input_fn(file_pattern: List[Text],
             data_accessor: tfx.components.DataAccessor,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int) -> tf.data.Dataset:
    """Generates features and label for tuning/training.
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=LABEL_COLUMN),
        tf_transform_output.transformed_metadata.schema).repeat()

def _get_json_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""

  model.tft_layer_inference = tf_transform_output.transform_features_layer()
  schema = tf_transform_output.raw_feature_spec()
  schema.pop(LABEL_COLUMN)

  feature_tensor_specs = list()
  for k,f in schema.items():
#         print(f"K => {k} f => {f}")
        feature_tensor_specs.append(tf.TensorSpec(shape=None, dtype=f.dtype, name=k))
  print(feature_tensor_specs)

  @tf.function(input_signature=feature_tensor_specs)
  def serve_json_fn(*args):
    """Returns the output to be used in the serving signature."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(LABEL_COLUMN)
    raw_features = {}
    index = 0
    for key in raw_feature_spec.keys():
        input_tensor = args[index]
        raw_features[key] = tf.sparse.from_dense(input_tensor)
#         print(f"key ============> {key} raw_features[key] ==========> {raw_features[key]}")
        index = index + 1
        
    transformed_features = model.tft_layer_inference(raw_features)
    
    for k,v in transformed_features.items():
        transformed_features[k] = tf.reshape(v, (1, v.shape[1]))

    outputs = model(transformed_features)
    return {'outputs': outputs}

  return serve_json_fn

def _get_tf_examples_serving_signature(model, tf_transform_output):
  """Returns a serving signature that accepts `tensorflow.Example`."""

  model.tft_layer_inference = tf_transform_output.transform_features_layer()
  schema = tf_transform_output.raw_metadata.schema
#   schema_dict = {}
#   for f in schema.feature:
#       if f.name != LABEL_COLUMN:
#         schema_dict[f.name] = tf.TensorSpec(shape=[None], dtype=f.type, name=f.name)
#   print(schema_dict)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
  def serve_tf_examples_fn(serialized_tf_example):
    """Returns the output to be used in the serving signature."""
    tf.print(serialized_tf_example)
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label feature since these will not be present at serving time.
    raw_feature_spec.pop(LABEL_COLUMN)
    raw_features = tf.io.parse_example(serialized_tf_example, features=raw_feature_spec)
    tf.print(raw_features)
    transformed_features = model.tft_layer_inference(raw_features)
    
    for k,v in transformed_features.items():
        transformed_features[k] = tf.reshape(v, (1, v.shape[1]))

    outputs = model(transformed_features)
#     raw_features['isfraud'] = outputs
    return {"output": outputs}

  return serve_tf_examples_fn

def _get_transform_features_signature(model, tf_transform_output):
  """Returns a serving signature that applies tf.Transform to features."""
  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
  model.tft_layer_eval = tf_transform_output.transform_features_layer()
    
  schema = tf_transform_output.raw_metadata.schema
  schema_dict = {}
  for f in schema.feature:
     if f.name != LABEL_COLUMN:
        schema_dict[f.name] = tf.TensorSpec(shape=[None], dtype=f.type, name=f.name)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
  def transform_features_fn(serialized_tf_example):
    """Returns the transformed_features to be fed as input to evaluator."""
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_features = tf.io.parse_example(serialized_tf_example, features=raw_feature_spec)
#     raw_features = serialized_tf_example
    transformed_features = model.tft_layer_eval(raw_features)
    tf.print('eval_transformed_features = %s', transformed_features)
    return transformed_features

  return transform_features_fn

def run_fn(fn_args: tfx.components.FnArgs):
    print(f"fn_args = {fn_args}")
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    schema = tf_transform_output.transformed_metadata.schema
    
    features = [f.name for f in schema.feature if f.name != LABEL_COLUMN]
    print(f"features = {features}")
#     model = FraudDetectDenseModel(features)
    
    inputs = [layers.Input(shape=(1,), name=f) for f in features]
    fd_graph = layers.concatenate(inputs)
    fd_graph = layers.Dense(256, activation='relu', name="dense_layer_1")(fd_graph)
    fd_graph = layers.Dense(256, activation='relu', name="dense_later_2")(fd_graph)
    fd_graph = layers.Dense(256, activation='relu', name="dense_layer_3")(fd_graph)
    fd_graph = layers.Dense(256, activation='relu', name="dense_later_4")(fd_graph)
    fd_graph = layers.Dropout(.1, name="dropout")(fd_graph)
    output = layers.Dense(1, activation='sigmoid', name="output")(fd_graph)
    
    model = keras.Model(inputs=inputs, outputs=output)
#     metrics.AUC(name='auc')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[metrics.AUC(name='auc')], loss=tf.keras.losses.BinaryCrossentropy())
    model.summary()
    
    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        TRAIN_BATCH_SIZE)

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        EVAL_BATCH_SIZE)
    
    EPOCHS = fn_args.custom_config.get('epochs', 1)
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)
    
    signatures = {
        'transform_features': _get_transform_features_signature(model, tf_transform_output),
        'serving_default': _get_tf_examples_serving_signature(model, tf_transform_output),
        'json_serving': _get_json_serving_signature(model, tf_transform_output)
    }
    model.save(fn_args.serving_model_dir, 
               save_format='tf', 
               signatures=signatures)
    
    return signatures
    