# THOR Challenge
http://vuchallenge.org/thor.html

## Requirements
* keras
* tensorflow
* scikit-image
* gym

## How to run
```
python a3c_atari.py
```

## How to run the A3C agent on custom environments
Please take a look at a3c_env/atari_env.py or a3c_env/cartpole_env.py to learn how to create your own environment. Then look at a3c_atari.py or a3c_cartpole.py to learn how to write your own script to train the A3C agent to play your environment.

## References
* [Mnih et al., 2016, Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [coreylynch/async-rl](https://github.com/coreylynch/async-rl)