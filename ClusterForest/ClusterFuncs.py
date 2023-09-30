# Cluster Functions for adding features to dataset
import sys
import copy
import numpy as np
from sklearn.neighbors import KernelDensity as kde
from sklearn.mixture import GaussianMixture as gm
from sklearn.ensemble import GradientBoostingClassifier as gbc
from .estimators import GenerousForest as gf
from . import Losses


class ClusterPred:
    def __init__(
        self,
        max_components=20,
        n_init=10,
        prob_thresh=0.9,
        cluster_iterations=10,
        bic_thresh=0,
        best_n=None,
        clf_spread=gf(),
        clf_pred=gf(),
    ):

        self.max_components = max_components
        self.n_init = n_init
        self.prob_thresh = prob_thresh
        self.cluster_iterations = cluster_iterations
        self.bic_thresh = bic_thresh
        self.best_n = best_n
        self.clf_spread = clf_spread
        self.clf_pred = clf_pred

    def gm_cluster_search(self, tree_error_outputs):
        """
        creates gaussian mixture model with given number of components. Trying this
        n_init times and keeping track of the best (lowest) BIC score.

        inputs:
                tree_error_outputs: nxd array - n=number of original features, d=number of trees in forest
                                                entries are errors for this row/tree
                max_clusters: the maximum number of components to consider for GMM
                n_init: how many initializations to try for each number of components

        returns:
                outmodel: the model with the lowest BIC
        """

        min_bic = sys.maxsize
        for i in range(1, self.max_components + 1):

            model = gm(n_init=self.n_init, n_components=i)
            model.fit(tree_error_outputs)
            bic = model.bic(tree_error_outputs)

            if bic < min_bic:

                outmodel = copy.deepcopy(model)
                min_bic = bic

        return outmodel, min_bic

    def gm_cluster_assign(self, model, tree_error_outputs):
        """
        assigns original features to the GMM component distribution to which they best match, if any
        outputs of this function will be used to train KDE over the datapoints for each cluster

        inputs:
                model - GMM model from gm_cluster_search that has best BIC
                tree_error_outputs: nxd array - n=number of original features, d=number of trees in forest
                                                entries are errors for this row/tree
                prob_thresh: float between 0 and 1 - the minimum normalized likelihood under some component
                                                    for this row to be assigned to that component

        returns:
                cluster_assignments: dict - indices of original features assigned to each cluster
                cluster_assignments: array
        """

        cluster_assignments = np.zeros(tree_error_outputs.shape[0])
        for i in range(tree_error_outputs.shape[0]):

            probs = model.predict_proba(tree_error_outputs[i].reshape(1, -1))[0]

            if max(probs) >= self.prob_thresh:

                cluster_assignments[i] = np.argmax(probs)

            else:
                cluster_assignments[i] = -1

        return cluster_assignments

    def reduce_to_important_features(
        self, features, cluster_assignments, important_features=None
    ):
        """
        for each GMM component assignment, reduce those rows to only the most iportant features for KDE

        inputs:
                features: nxd array of original features
                cluster_assignments: dict - output of gm_cluster_assign
                important_features: 1xd array of feature indices on which to build KDE

        returns:
                cluster_assignments: dict - key: component_number, value: rows of data (only important indices)
        """

        if important_features is None:
            important_features = list(range(len(features[0])))

        for i in cluster_assignments:

            new = list()
            for j in cluster_assignments[i]:

                new.append(np.array(features[j])[important_features])

            cluster_assignments[i] = np.array(new)

        return cluster_assignments

    def predict_cluster_assignment_train(self, features, cluster_assignments, clf_pred):
        """
        train xgb model for cluster assignments

        """
        clf_pred.fit(features, cluster_assignments)

    def kde_by_cluster(cluster_assignments, kernel="gaussian", bandwidth="scott"):
        """
        create KDE for each mixture component

        inputs:
                cluster_assignments: dict - output of reduce to important features
                kernel: str - kernel to use for KDE
                bandwidth: float or str - bandwidth to use for KDE

        returns:
                cluster_densities: dict -kde distribution for each GMM component
        """

        cluster_densities = dict()
        for i in cluster_assignments:

            model = kde(kernel=kernel, bandwidth=bandwidth)
            model.fit(cluster_assignments[i])

            cluster_densities[i] = model

        return cluster_densities

    def assign_kde_scores(features, cluster_densities, important_features=None):
        """
        assign each row a log density under each component KDE distribution

        inputs:
                features: array of original features
                cluster_densities: dict - output of kde_by_cluster
                important_features: features on which the KDE models are constructed

        returns:

        """

        if important_features is None:
            important_features = list(range(features.shape[1]))

        densities = np.zeros((len(features), len(cluster_densities)))
        for i in range(features.shape[0]):

            row = np.array(features[i])[important_features]

            for j in cluster_densities:
                densities[i, j] = cluster_densities[j].score(row.reshape(1, -1))

        return densities

    def get_important_features(clf, n_features):
        """
        get the n most important features on which to construct KDE

        inputs:
                clf - generous forest classifier
                n_features: int - number of features to select

        returns:
                array of len n_features containing indices of most important features
        """

        return np.argsort(clf.feature_importances)[::-1][:n_features]

    def row_tree_errors(self, features, labels, clf_spread):
        """
        create nxd array where entries are error for that particular tree and row

        inputs:
                clf: generous forest classifier
                features: nxd features array
                labels: dx1 array of labels
                loss: loss func that takes node as input

        returns:
                num feature rows x num trees array of errors
        """

        # create larger array to hold all results by tree
        tree_results = np.zeros((len(labels), len(clf_spread.trees)))

        for j in range(len(self.clf_spread.trees)):

            preds = self.clf_spread.trees[j].predict(features, return_node=True)

            for i in range(len(preds)):

                tree_results[i, j] = self.loss(
                    preds[i], self.clf_spread.label_map_train[labels[i]]
                )

        return tree_results

    @staticmethod
    def pred_correlation(tree_results):

        res = 0
        comps = 0

        for i in range(len(tree_results)):
            for j in range(len(tree_results)):

                if j > i:

                    res += sum(abs(tree_results[i] - tree_results[j])) / len(
                        tree_results[i]
                    )
                    comps += 1

        res = res / comps
        return res

    def create_models_clf(self, features, labels, clf_spread=None, clf_pred=None):
        """
        string together functions related to clustering that make use of XGB
        """
        if clf_spread is None:
            clf_spread = self.clf_spread

        if clf_pred is None:
            clf_pred = self.clf_pred

        tree_outputs = self.row_tree_errors(features, labels, clf_spread)
        model, min_bic = self.gm_cluster_search(tree_outputs)
        cluster_assignments = self.gm_cluster_assign(model, tree_outputs)
        self.predict_cluster_assignment_train(clf_pred, features, cluster_assignments)
        # clf_pred_score = clf_pred.score(features, cluster_assignments)

        return (model, min_bic)

    def add_features_sequential(self, features, model_list):
        """
        sequentially add cluster features to original feature set
        """
        new_feature_array = list()
        for i in model_list:

            new_feature_array.append(i.predict_proba(features))

        return np.concatenate([features] + new_feature_array, axis=1)

    def create_new_features_sequential(self, features, labels, improve_select=False):

        additive_models = list()
        bic_to_beat = sys.maxsize

        for _ in range(self.cluster_iterations):

            # start by fitting original model
            clf_spread_ = copy.deepcopy(self.clf_spread)
            clf_pred_ = copy.deepcopy(self.clf_pred)
            clf_spread_.fit(features, labels)
            (_, min_bic) = self.create_models_clf(
                features, labels, clf_spread_, clf_pred_
            )

            if min_bic < bic_to_beat:

                features = self.add_features_sequential(features, [clf_pred_])
                additive_models.append(clf_pred_)

                if not improve_select:
                    bic_to_beat = sys.maxsize

                else:
                    bic_to_beat = min_bic

        return (features, additive_models)

    def create_new_features_best_n(self, features, labels):

        additive_models = list()
        bics = list()
        if self.bic_thresh is None:
            bic_thresh = sys.maxsize

        else:
            bic_thresh = self.bic_thresh

        for _ in range(self.cluster_iterations):

            # start by fitting original model
            clf_spread_ = copy.deepcopy(self.clf_spread)
            clf_pred_ = copy.deepcopy(self.clf_pred)
            clf_spread_.fit(features, labels)
            (_, min_bic) = self.create_models_clf(
                features, labels, clf_spread_, clf_pred_
            )

            # first verify that we are less than bic thresh
            if min_bic < bic_thresh:

                additive_models.append(clf_pred_)
                bics.append(min_bic)

        additive_models = np.array(additive_models)
        bics = np.array(bics)

        indices = np.argsort(bics)
        additive_models = additive_models[indices][: self.best_n]

        features = self.add_features_sequential

        return (features, additive_models)
