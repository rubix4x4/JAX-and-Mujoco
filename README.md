Folder initially was to explore the use of JAX and Mujoco, only to find that the XLA and MJX support isn't there for my system. so the JAX folder has a NN example using the FLAX system, but will probably remain uneddited for now

RL Folder:
Reinforcement learning, samples plus custom implementation. Using Mujoco to simulate the movement of Skydio x2 drone, and using reinforcement learning (Stable Baselines 3 SAC Policy). Learning to have it hover about 2m in the air. Testing out different reward functions, then maybe will adjust hyperparameters of SAC model.

Current Status: Successfully trained a drone that hovers with rough stability around the desired location. Next steps will be to have the system randomize the starting location and the desired hover point during training to prevent the system from overfitting the specific start and stop conditions.
