#+TITLE: An implementation of MADDPG
#+AUTHOR: xuehy
#+EMAIL: hyxue@outlook.com
#+STARTUP: content

* 1. Introduction

This is a pytorch implementation of [[https://arxiv.org/abs/1706.02275][multi-agent deep deterministic policy gradient algorithm]].

The experimental environment is a modified version of Waterworld based on [[https://github.com/sisl/MADRL][MADRL]]. 

* 2. Environment

The main features (different from MADRL) of the modified Waterworld environment are:

- evaders and poisons now bounce at the wall obeying physical rules
- sizes of the evaders, pursuers and poisons are now the same so that random actions will lead to average rewards around 0.
- need exactly n_coop agents to catch food.

* 3. Dependency

- [[https://github.com/pytorch/pytorch][pytorch]]
- [[https://github.com/facebookresearch/visdom][visdom]]
- =python==3.6.1= (recommend using the anaconda/miniconda)
- if you need to render the environments, =opencv= is required

* 4. Install

- Install [[https://github.com/sisl/MADRL][MADRL]].
- Replace the =madrl_environments/pursuit= directory with the one in this repo.
- =python main.py=

if scene rendering is enabled, recommend to install =opencv= through [[https://github.com/conda-forge/opencv-feedstock][conda-forge]].

* 5. Results

** two agents, cooperation = 2
The two agents need to cooperate to achieve the food for reward 10.

[[PNG/demo.gif]]

[[PNG/3.png]]

the average

[[PNG/4.png]]

** one agent, cooperation = 1

[[PNG/newplot.png]]


* 6. TODO

- reproduce the experiments in the paper with competitive environments.
