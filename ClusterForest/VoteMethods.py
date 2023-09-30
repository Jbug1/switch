import numpy as np
from collections import Counter

# implement different voting systems...tree_error, leaf_error, combined_error, ranked choice?...put these in a class


def majority_vote(preds):

    outs = np.empty(preds.shape[0], dtype=object)

    for i in range(preds.shape[0]):

        votes = Counter(preds[i])
        outs[i] = max(votes, key=votes.get)

    return outs


def tree_weight_vote(preds, tree_probs, probabilities):

    outs = np.empty(preds.shape[0], dtype=object)

    for i in range(preds.shape[0]):
        pred_dict = dict()

        for j in range(preds.shape[1]):
            if preds[i, j] in pred_dict:
                pred_dict[preds[i, j]] += tree_probs[j]

        else:
            pred_dict[preds[i, j]] = tree_probs[j]

        outs[i] = max(pred_dict, key=pred_dict.get)

    return outs


def prob_weight_vote(preds, tree_probs, probabilities):

    outs = np.empty(preds.shape[0], dtype=object)

    for i in range(preds.shape[0]):
        pred_dict = dict()

        for j in range(preds.shape[1]):
            if preds[i, j] in pred_dict:
                pred_dict[preds[i, j]] += probabilities[i, j]

        else:
            pred_dict[preds[i, j]] = probabilities[i, j]

        outs[i] = max(pred_dict, key=pred_dict.get)

    return outs
