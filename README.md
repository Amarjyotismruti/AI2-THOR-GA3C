# THOR Challenge
http://vuchallenge.org/thor.html

## Requirements
* python 3.4.2
* keras 2.0.2
* tensorflow 1.0.1
* scikit-image 0.13.0
* gym 0.8.1

## How to run
```
python a3c_atari.py
```

## How to run the A3C agent on custom environments
Please take a look at a3c_env/atari_env.py or a3c_env/cartpole_env.py to learn how to create your own environment. Then look at a3c_atari.py or a3c_cartpole.py to learn how to write your own script to train the A3C agent to play your environment.

## Note
Go into lib/python3.4/site-packages/robosims/controller.py and change 'from queue import Queue' to 'from multiprocessing import Queue' for GA3C to work with THOR

## References
* [Mnih et al., 2016, Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
* [coreylynch/async-rl](https://github.com/coreylynch/async-rl)