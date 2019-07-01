# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""

from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time

np.random.seed(42)
tf.logging.set_verbosity(tf.logging.ERROR)

class Model(object):
  """Fully connected neural network with no hidden layer."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False

    self.metadata = metadata
    self.output_dim = self.metadata.get_output_size()

    # Set batch size (for both training and testing)
    self.batch_size = 30

    # Get model function from class method below
    model_fn = self.model_fn
    # Change to True if you want to show device info at each operation
    log_device_placement = False
    session_config = tf.ConfigProto(log_device_placement=log_device_placement)
    # Classifier using model_fn (see below)
    self.classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      config=tf.estimator.RunConfig(session_config=session_config))

    # Attributes for preprocessing
    self.default_image_size = (112,112)
    self.default_num_frames = 10
    self.default_shuffle_buffer = 100

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.train_begin_times = []
    self.test_begin_times = []
    self.li_steps_to_train = []
    self.li_cycle_length = []
    self.li_estimated_time = []
    self.time_estimator = LinearRegression()
    # Critical number for early stopping
    # Depends on number of classes (output_dim)
    # see the function self.choose_to_stop_early() below for more details
    self.num_epochs_we_want_to_train = 1

  def train(self, dataset, remaining_time_budget=None):
    """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
    if self.done_training:
      return

    # Count examples on training set
    if not hasattr(self, 'num_examples_train'):
      logger.info("Counting number of examples on train set.")
      iterator = dataset.make_one_shot_iterator()
      example, labels = iterator.get_next()
      sample_count = 0
      with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        while True:
          try:
            sess.run(labels)
            sample_count += 1
          except tf.errors.OutOfRangeError:
            break
      self.num_examples_train = sample_count
      logger.info("Finished counting. There are {} examples for training set."\
                  .format(sample_count))

    self.train_begin_times.append(time.time())
    if len(self.train_begin_times) >= 2:
      cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
      self.li_cycle_length.append(cycle_length)

    # Get number of steps to train according to some strategy
    steps_to_train = self.get_steps_to_train(remaining_time_budget)

    if steps_to_train <= 0:
      logger.info("Not enough time remaining for training + test. " +
                "Skipping training...")
      self.done_training = True
    elif self.choose_to_stop_early():
      logger.info("The model chooses to stop further training because " +
                  "The preset maximum number of epochs for training is " +
                  "obtained: self.num_epochs_we_want_to_train = " +
                  str(self.num_epochs_we_want_to_train))
      self.done_training = True
    else:
      msg_est = ""
      if len(self.li_estimated_time) > 0:
        estimated_duration = self.li_estimated_time[-1]
        estimated_end_time = time.ctime(int(time.time() + estimated_duration))
        msg_est = "estimated time for training + test: " +\
                  "{:.2f} sec, ".format(estimated_duration)
        msg_est += "and should finish around {}.".format(estimated_end_time)
      logger.info("Begin training for another {} steps...{}"\
                .format(steps_to_train, msg_est))

      # Prepare input function for training
      train_input_fn = lambda: self.input_function(dataset, is_training=True)

      # Start training
      train_start = time.time()
      self.classifier.train(input_fn=train_input_fn, steps=steps_to_train)
      train_end = time.time()

      # Update for time budget managing
      train_duration = train_end - train_start
      self.li_steps_to_train.append(steps_to_train)
      logger.info("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
            "Now total steps trained: {}. ".format(sum(self.li_steps_to_train)) +\
            "Total time used for training + test: {:.2f} sec. ".format(sum(self.li_cycle_length)))

  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
    # Count examples on test set
    if not hasattr(self, 'num_examples_test'):
      logger.info("Counting number of examples on test set.")
      iterator = dataset.make_one_shot_iterator()
      example, labels = iterator.get_next()
      sample_count = 0
      with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        while True:
          try:
            sess.run(labels)
            sample_count += 1
          except tf.errors.OutOfRangeError:
            break
      self.num_examples_test = sample_count
      logger.info("Finished counting. There are {} examples for test set."\
                  .format(sample_count))

    test_begin = time.time()
    self.test_begin_times.append(test_begin)
    logger.info("Begin testing...")

    # Prepare input function for testing
    test_input_fn = lambda: self.input_function(dataset, is_training=False)

    # Start testing (i.e. making prediction on test set)
    test_results = self.classifier.predict(input_fn=test_input_fn)

    predictions = [x['probabilities'] for x in test_results]
    predictions = np.array(predictions)
    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    logger.info("[+] Successfully made one prediction. {:.2f} sec used. "\
              .format(test_duration) +\
              "Duration used for test: {:2f}".format(test_duration))
    return predictions

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  # Model functions that contain info on neural network architectures
  # Several model functions are to be implemented, for different domains
  def model_fn(self, features, labels, mode):
    """Auto-Scaling 3D CNN model.

    For more information on how to write a model function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
    """
    input_layer = features

    # Replace missing values by 0
    hidden_layer = tf.where(tf.is_nan(input_layer),
                           tf.zeros_like(input_layer), input_layer)

    # Sum over time axis
    hidden_layer = tf.reduce_sum(hidden_layer, axis=1)

    # Flatten
    hidden_layer = tf.layers.flatten(hidden_layer)

    logits = tf.layers.dense(inputs=hidden_layer, units=self.output_dim)
    sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # "classes": binary_predictions,
      # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": sigmoid_tensor
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # For multi-label classification, a correct loss is sigmoid cross entropy
    loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    assert mode == tf.estimator.ModeKeys.EVAL
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def input_function(self, dataset, is_training):
    """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
    """
    dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))

    if is_training:
      # Shuffle input examples
      dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
      # Convert to RepeatDataset to train for several epochs
      dataset = dataset.repeat()

    # Set batch size
    dataset = dataset.batch(batch_size=self.batch_size)

    iterator_name = 'iterator_train' if is_training else 'iterator_test'

    if not hasattr(self, iterator_name):
      self.iterator = dataset.make_one_shot_iterator()

    # iterator = dataset.make_one_shot_iterator()
    iterator = self.iterator
    example, labels = iterator.get_next()
    return example, labels

  def preprocess_tensor_4d(self, tensor_4d):
    """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
    non-fixed). The output tensor wil have fixed, known shape.

    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels]
          where some dimensions might be `None`.
    Returns:
      A 4-D Tensor with fixed, known shape.
    """
    tensor_4d_shape = tensor_4d.shape
    logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

    if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
      num_frames = tensor_4d_shape[0]
    else:
      num_frames = self.default_num_frames
    if tensor_4d_shape[1] > 0:
      new_row_count = tensor_4d_shape[1]
    else:
      new_row_count=self.default_image_size[0]
    if tensor_4d_shape[2] > 0:
      new_col_count = tensor_4d_shape[2]
    else:
      new_col_count=self.default_image_size[1]

    if not tensor_4d_shape[0] > 0:
      logger.info("Detected that examples have variable sequence_size, will " +
                "randomly crop a sequence with num_frames = " +
                "{}".format(num_frames))
      tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
    if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
      logger.info("Detected that examples have variable space size, will " +
                "resize space axes to (new_row_count, new_col_count) = " +
                "{}".format((new_row_count, new_col_count)))
      tensor_4d = resize_space_axes(tensor_4d,
                                    new_row_count=new_row_count,
                                    new_col_count=new_col_count)
    logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
    return tensor_4d

  def get_steps_to_train(self, remaining_time_budget):
    """Get number of steps for training according to `remaining_time_budget`.

    The strategy is:
      1. If no training is done before, train for 10 steps (ten batches);
      2. Otherwise, double the number of steps to train. Estimate the time
         needed for training and test for this number of steps;
      3. Compare to remaining time budget. If not enough, stop. Otherwise,
         proceed to training/test and go to step 2.
    """
    if remaining_time_budget is None: # This is never true in the competition anyway
      remaining_time_budget = 1200 # if no time limit is given, set to 20min

    # for more conservative estimation
    remaining_time_budget = min(remaining_time_budget - 60,
                                remaining_time_budget * 0.95)

    if len(self.li_steps_to_train) == 0:
      return 10
    else:
      steps_to_train = self.li_steps_to_train[-1] * 2

      # Estimate required time using linear regression
      X = np.array(self.li_steps_to_train).reshape(-1, 1)
      Y = np.array(self.li_cycle_length)
      self.time_estimator.fit(X, Y)
      X_test = np.array([steps_to_train]).reshape(-1, 1)
      Y_pred = self.time_estimator.predict(X_test)

      estimated_time = Y_pred[0]
      self.li_estimated_time.append(estimated_time)

      if estimated_time >= remaining_time_budget:
        return 0
      else:
        return steps_to_train

  def age(self):
    return time.time() - self.birthday

  def choose_to_stop_early(self):
    """The criterion to stop further training (thus finish train/predict
    process).
    """
    batch_size = self.batch_size
    num_examples = self.num_examples_train
    num_epochs = sum(self.li_steps_to_train) * batch_size / num_examples
    logger.info("Model already trained for {:.4f} epochs.".format(num_epochs))
    return num_epochs > self.num_epochs_we_want_to_train # Train for at least certain number of epochs then stop

def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
  """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
  labels = tf.cast(labels, dtype=tf.float32)
  relu_logits = tf.nn.relu(logits)
  exp_logits = tf.exp(- tf.abs(logits))
  sigmoid_logits = tf.log(1 + exp_logits)
  element_wise_xent = relu_logits - labels * logits + sigmoid_logits
  return tf.reduce_sum(element_wise_xent)

def get_num_entries(tensor):
  """Return number of entries for a TensorFlow tensor.

  Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
  Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
  """
  tensor_shape = tensor.shape
  assert(len(tensor_shape) > 1)
  num_entries  = 1
  for i in tensor_shape[1:]:
    num_entries *= int(i)
  return num_entries

def crop_time_axis(tensor_4d, num_frames, begin_index=None):
  """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[0], 0)
  padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[0] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1, -1])

  return sliced_tensor

def resize_space_axes(tensor_4d, new_row_count, new_col_count):
  """Given a 4-D tensor, resize space axes to have target size.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
  resized_images = tf.image.resize_images(tensor_4d,
                                          size=(new_row_count, new_col_count))
  return resized_images

def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')
