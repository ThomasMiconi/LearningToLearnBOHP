# Learning to learn with Backpropagation of Hebbian Plasticity

This is the source code for the experiments described in the Arxiv preprint, [Learning to learn with backpropagation of Hebbian Plasticity](https://arxiv.org/abs/1609.02228).

There are three directories, one per experiment:

* `completion/completion.py`: the pattern completion experiment.
* `oneshot/oneshot.py`: the one-shot learning experiment.
* `reversal/reversal.py`: the reversal learning experiment.

The BOHP algorithm optimizes both the weight and the plasticity of all
connections, so that during training, the network "learns how to learn" the relevant associations through Hebbian plasticity during each episode.

You can run the scripts as-is, or modify some parameters with command-line arguments. Default parameters are the ones used in the paper.
The program will store the mean absolute errors of each episode in errs.txt, as well as other data (weights, etc.) in data.pkl. See code for details.

Example (from IPython):

    cd oneshot
    %run oneshot.py LEARNPERIOD 2 NBITER 20 YSIZE 2 ALPHATRACE .98

