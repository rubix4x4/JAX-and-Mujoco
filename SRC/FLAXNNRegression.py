import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from jax import random
import numpy as np



# Define the model
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=10)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# Define a function to create the initial parameters
def create_train_state(rng, learning_rate):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 2]))['params']
    tx = optax.sgd(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the loss function
def mse_loss(params, batch):
    inputs, targets = batch
    predictions = MLP().apply({'params': params}, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Define the training step
@jax.jit
def train_step(state, batch):
    grads = jax.grad(mse_loss)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state



print(jax.default_backend)



# Create the data
key = random.PRNGKey(0)
x = jnp.array(np.random.randn(100, 2))  # 100 samples, 2 features
y = jnp.array(np.random.randn(100, 1))  # 100 samples, 1 target
batch = (x, y)

# Initialize training state
learning_rate = 0.01
state = create_train_state(key, learning_rate)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    state = train_step(state, batch)
    if epoch % 100 == 0:
        loss_value = mse_loss(state.params, batch)
        print(f'Epoch {epoch}, Loss: {loss_value}')

# Test the model
test_x = jnp.array(np.random.randn(10, 2))
preds = MLP().apply({'params': state.params}, test_x)
print(preds)
