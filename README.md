Folder initially was to explore the use of JAX and Mujoco, only to find that the XLA and MJX support isn't there for my system. so the JAX folder has a NN example using the FLAX system, but will probably remain uneddited for now

RL Folder:
Reinforcement learning, samples plus custom implementation. Using Mujoco to simulate the movement of Skydio x2 drone, and using reinforcement learning to have it hover about 2m in the air. Testing out different reward functions, then maybe will adjust hyperparameters of SAC model.
    Current best approach was to reward getting close to the location, and punish large velocities, but after a long amount of time, it seems to have decided that the best policy is to simply lay down and do nothing.

    Will try penalizing distance and rewarding velocity similarity to target.
