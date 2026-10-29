"""
Discrete Hidden Markov Model algorithms: Forward and Viterbi.
This is a reusable educational implementation; parameters must be estimated
from labelled/sequenced data before interpreting it as a customer model.
"""
import numpy as np


def forward_probability(start_prob, transition_prob, emission_prob, observations):
    """Return the probability of an observation sequence using Forward DP."""
    start = np.asarray(start_prob, dtype=float)
    trans = np.asarray(transition_prob, dtype=float)
    emit = np.asarray(emission_prob, dtype=float)
    obs = list(observations)
    if not obs:
        raise ValueError("observations must contain at least one symbol")
    if np.any(start < 0) or np.any(trans < 0) or np.any(emit < 0):
        raise ValueError("Probabilities cannot be negative")
    alpha = start * emit[:, obs[0]]
    for symbol in obs[1:]:
        alpha = (alpha @ trans) * emit[:, symbol]
    return float(alpha.sum())


def viterbi_decode(start_prob, transition_prob, emission_prob, observations):
    """Return the most likely hidden-state index path and its probability."""
    start = np.asarray(start_prob, dtype=float)
    trans = np.asarray(transition_prob, dtype=float)
    emit = np.asarray(emission_prob, dtype=float)
    obs = list(observations)
    if not obs:
        raise ValueError("observations must contain at least one symbol")
    # Log probabilities avoid underflow for longer sequences.
    with np.errstate(divide="ignore"):
        log_start = np.log(start)
        log_trans = np.log(trans)
        log_emit = np.log(emit)
    delta = log_start + log_emit[:, obs[0]]
    backpointers = []
    for symbol in obs[1:]:
        candidates = delta[:, None] + log_trans
        back = np.argmax(candidates, axis=0)
        delta = candidates[back, np.arange(len(start))] + log_emit[:, symbol]
        backpointers.append(back)
    last_state = int(np.argmax(delta))
    path = [last_state]
    for back in reversed(backpointers):
        path.append(int(back[path[-1]]))
    path.reverse()
    probability = float(np.exp(np.max(delta)))
    return path, probability


if __name__ == "__main__":
    # Example only: two hidden states and two observation symbols.
    start = [0.6, 0.4]
    transition = [[0.7, 0.3], [0.4, 0.6]]
    emission = [[0.5, 0.5], [0.1, 0.9]]
    sequence = [0, 1, 0, 1]  # Example symbols, not real customer labels.
    print("Observation-sequence probability:", forward_probability(
        start, transition, emission, sequence))
    path, probability = viterbi_decode(start, transition, emission, sequence)
    print("Most likely hidden-state path:", path)
    print("Viterbi path probability:", probability)
