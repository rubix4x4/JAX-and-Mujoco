# This is me pulling a mostly pre-made tutorial on convolutional neural networks to understand jax NN structure.
# Custom commenting for my own understanding of system structure and as I look through documentation to understand jnp function syntax
import jax
import jax.numpy as jnp
import optax
import flax
from jax import grad, jit, random
from flax import linen as nn
from flax.training import train_state
import numpy as np
import tensorflow_datasets as tfds

class CNN(nn.Module):
  """A simple CNN model."""
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)           # 3x3 kernel, features refers to the number of convolution filters
    x = nn.relu(x)                                            # Rectified linear unit activation
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))   # Pool value by average, reduces pixel size to 14x14
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))   # Same pooling, reduces down to 7 by 7
    x = x.reshape((x.shape[0], -1))                           # flatten, -1 is to inherit size
    x = nn.Dense(features=256)(x)                             # Fully connected layer, 256 outputs
    x = nn.relu(x)                                            # Rectified Linear unit activation
    x = nn.Dense(features=10)(x)                              # Outputs the likelihood of different being different features, 10 total labels
    return x

def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes = 10)  # One Hot classification for distinct labels where number does not correspond to any real value
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits = logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1 )==labels)      # Checks if the highest probability classification matches label, and then averages the binaries to identify average accuracy
    metrics = {
        'loss':loss,
        'accuracy':accuracy,
    }
    return metrics

def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.   # Normalizes input data (Black and White image)
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.     # Normalizes input data (Black and White image)
  return train_ds, test_ds
    
def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = CNN() # creates our neural network
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']  # 1 input, 28 by 28 pixels, 1 color channel
  tx = optax.sgd(learning_rate, momentum)                    # Stochastic Gradient Descent Optimizer
  #tx = optax.adam(learning_rate_adam)                         # Try Adam Solver
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)
  
@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits=logits, labels=batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=batch['label'])
  return state, metrics

@jax.jit
def eval_step(params, batch):
  logits = CNN().apply({'params': params}, batch['image'])
  return compute_metrics(logits=logits, labels=batch['label'])

def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size
  
  # This section shapes each batch in the epoch
  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []
  for perm in perms:
    batch = {k: v[perm, ...] for k, v in train_ds.items()}
    state, metrics = train_step(state, batch)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state

def eval_model(params, test_ds):
  metrics = eval_step(params, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']

# Pulls data set
train_ds, test_ds = get_datasets()

rng = jax.random.PRNGKey(0)                   # create RNG sets
rng, init_rng = jax.random.split(rng)

learning_rate = 0.1                           # parameters of the stochastic gradient descent optimizer
learning_rate_adam = 0.001
momentum = 0.9

state = create_train_state(init_rng, learning_rate, momentum)
del init_rng  # Must not be used anymore.

num_epochs = 5
batch_size = 32

for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  # Evaluate on the test set after each training epoch 
  test_loss, test_accuracy = eval_model(state.params, test_ds)
  print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
      epoch, test_loss, test_accuracy * 100))