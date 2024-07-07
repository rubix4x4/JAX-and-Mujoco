# Adapt a CNN network for use on the oxford_flower102 dataset from tensorflow
import jax
import jax.numpy as jnp
import optax
import flax
from jax import grad, jit, random
from flax import linen as nn
from flax.training import train_state
import numpy as np
import tensorflow_datasets as tfds
import time

class CNN(nn.Module):
  """A simple CNN model."""
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)           # 3x3 kernel, features refers to the number of convolution filters
    x = nn.relu(x)                                            # Rectified linear unit activation
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))   # Pool value by average
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))   
    x = nn.Conv(features=256, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))   
    x = nn.Conv(features=512, kernel_size=(3, 3))(x)
    x = nn.relu(x)    
    x = x.reshape((x.shape[0], -1))                           # flatten, -1 is to inherit size, needed to add more conv and pools to make this manageable
    x = nn.Dense(features=256)(x)                             # Fully connected layer, 256 outputs
    x = nn.relu(x)                                            # Rectified Linear unit activation
    x = nn.Dense(features=102)(x)                             # Outputs the likelihood of different being different features, 102 total labels
    return x

def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes = 102)  # One Hot classification for 102 distint flower types.
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
  """Load datasets into memory."""
  ds_builder = tfds.builder('oxford_flowers102')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1)) # 1020 images, 993 by 919, 3 channels
  # test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1)) 
  validate_ds = tfds.as_numpy(ds_builder.as_dataset(split='validation', batch_size = -1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.   # Normalizes input data
  # test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  validate_ds['image'] = jnp.float32(validate_ds['image'])/255.
  
  # Delete filename attribute in _ds dict
  train_ds.pop('file_name')
  validate_ds.pop('file_name')
  
  return train_ds, validate_ds
    
def create_train_state(rng, learning_rate, momentum):
  """Creates initial `TrainState`."""
  cnn = CNN() # creates our neural network
  params = cnn.init(rng, jnp.ones([1, 993, 919, 3]))['params']  # 1 initial image, 993 by 919 pixels, 3 channels
  #tx = optax.sgd(learning_rate, momentum)                    # Stochastic Gradient Descent Optimizer
  tx = optax.adam(learning_rate)                         # Try Adam Solver
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
    start = time.time()
    state, metrics = train_step(state, batch)
    end = time.time()
    length= end-start
    print('Batch Training Time = ' ,length)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
      epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

  return state

def eval_model(params, validate_ds):
  metrics = eval_step(params, validate_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']


print(jax.default_backend)


# Pulls data set
start = time.time()
train_ds, validate_ds = get_datasets()
end = time.time()
length = end-start
print('Data loading took ',length,' seconds')

rng = jax.random.PRNGKey(0)                   # create RNG sets
rng, init_rng = jax.random.split(rng)
#learning_rate = 0.1                           # parameters of the stochastic gradient descent optimizer
learning_rate_adam = 0.001
momentum = 0.9
state = create_train_state(init_rng, learning_rate_adam, momentum)
del init_rng  # Must not be used anymore.
num_epochs = 5
batch_size = 64

print('Training Start')
for epoch in range(1, num_epochs + 1):
  # Use a separate PRNG key to permute image data during shuffling
  rng, input_rng = jax.random.split(rng)
  # Run an optimization step over a training batch
  
  start = time.time()
  state = train_epoch(state, train_ds, batch_size, epoch, input_rng)
  end = time.time()
  length = end-start
  
  # Evaluate on the test set after each training epoch 
  test_loss, test_accuracy = eval_model(state.params, validate_ds)
  print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
      epoch, test_loss, test_accuracy * 100))
  print(' Time elapsed = ',length,' seconds')