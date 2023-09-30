from .GenerousNode import GenerousNode
from . import Transform
from . import Losses
import numpy as np
from collections import Counter
import sys
from scipy.stats import entropy
from scipy.stats import median_abs_deviation as mad


class GenerousTree:
    """
    Decision tree that implements random selection according to ratio of purity metrics
    """

    def __init__(
        self,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        features_cleaned=False,
        transform_method=Transform.sigmoid,
        best_n=None,
        scaling=1,
        label_map_train=None,
        label_map_pred=None,
        slots=0,
        max_features=None,
        bootstrap=True,
        root=None,
        leaves=0,
        nodes=0,
        max_iter=1e4,
        early_stop_thresh=0.95,
        lambda_=0.001,
        depth_scale=False,
        pb_block_size=1e4,
        pb_loss=Losses.zero_one_loss,
        eval_loss=Losses.zero_one_loss_eval,
        size_scale=False,
        pb_uniform_overlay=0,
        forest=None,
        max_depth=None,
        best_n_scale=None,
        boundary_multiplier=1,
        ovr_cycles=5,
    ):

        # user specified attributes
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.features_cleaned = features_cleaned
        self.transform_method = transform_method
        self.best_n = best_n
        self.scaling = scaling
        self.slots = slots
        self.label_map_train = label_map_train
        self.label_map_pred = label_map_pred
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.root = root
        self.leaves = leaves
        self.nodes = nodes
        self.max_iter = int(max_iter)
        self.early_stop_thresh = early_stop_thresh
        self.lambda_ = lambda_
        self.depth_scale = depth_scale
        self.pb_block_size = pb_block_size
        self.pb_loss = pb_loss
        self.size_scale = size_scale
        self.pb_uniform_overlay = pb_uniform_overlay
        self.eval_loss = eval_loss
        self.forest = forest
        self.max_depth = max_depth
        self.best_n_scale = best_n_scale
        self.boundary_multiplier = boundary_multiplier
        self.ovr_cycles = ovr_cycles

        # result attributes
        self.total_features = None
        self.leaves_ = list()
        self.max_features_ = None
        self.train_error = 0
        self.total_samples = 0
        self.pb_score = 0
        self.early_stop = False
        self.split_features_changed = 0
        self.split_points_changed = 0
        self.feature_importances = None

    def split_feature_aggregate_ovr(self, features, labels):

        if self.max_features is None:
            # consider all features
            feature_subset = np.arange(features.shape[1])
            impurities = np.zeros(features.shape[1])
            split_points = np.zeros(features.shape[1])

        else:
            # pick random subset of features
            feature_subset = np.random.choice(
                features.shape[1], size=self.max_features, replace=False
            )

            impurities = np.zeros(self.max_features)
            split_points = np.zeros(self.max_features)

        y_vals = set(labels)

        for i in feature_subset:
            boundary_mad = mad(features[:, i])

            # keep the best zero_one_error split
            min_error = sys.maxsize
            min_split_point = 0
            for j in y_vals:

                mask = np.where(
                    labels == j, np.ones(len(labels)), False
                )  # creates array with values 1 and 0 only

                error, split_point = self.nudge_split(
                    features[:, i], mask, boundary_mad
                )

                if error < min_error:
                    min_error = error
                    min_split_point = split_point

            impurities[i] = min_error
            split_points[i] = min_split_point

        sort_order = np.argsort(impurities)
        feature_subset = feature_subset[sort_order]

        impurities = impurities[sort_order]
        split_points = split_points[sort_order]

        return (feature_subset, impurities, split_points)

    def ovr_fit(self, features, labels):

        # print(f"lksdjflkdjsfl {len(features)}")
        self.total_samples = len(labels)
        self.total_features = features.shape[1]
        self.feature_importances = np.zeros(self.total_features)

        # check to see if we have generated slots and label map
        if self.label_map_train is None:

            (
                self.slots,
                self.label_map_train,
                self.label_map_pred,
            ) = GenerousTree.preprocess_labels(labels)

            self.labels_processed = True
            self.total_samples = len(labels)

        if type(self.max_features) != int and self.max_features is not None:

            self.get_max_features(features)

        # only create the root on the first split, otherwise we will have passed a non-null node
        self.nodes += 1
        self.root = GenerousNode(parent=None, depth=1)

        # call fit_inner to grow tree
        self.ovr_fit_inner(features, labels, cur_node=self.root)

    def ovr_fit_inner(self, features, labels, cur_node):

        # update cur_node
        cur_node.impurity = self.check_node_impurity(labels)
        cur_node.num_instances = len(labels)

        # check stop conditions for recursion, generate probabilities if so
        if (
            len(features) < self.min_samples_split
            or cur_node.impurity == 0
            or cur_node.depth == self.max_depth
        ):

            cur_node.probabilities = self.generate_probabilities(labels)
            cur_node.pred = np.argmax(cur_node.probabilities)
            cur_node.pred_prob = max(cur_node.probabilities)

            self.train_error += (
                (1 - cur_node.pred_prob) * len(labels) / self.total_samples
            )
            self.leaves += 1
            self.leaves_.append(cur_node)

            return

        # otherwise continue fitting the tree in ovr fashion
        (
            cur_node.split_features,
            impurities,
            cur_node.split_points,
        ) = self.split_feature_aggregate_ovr(features, labels)

        # select the split feature by applying chosen probability transform
        (
            split_feature_index,
            cur_node.split_prob,
            cur_node.feature_probs,
        ) = self.split_feature_select(impurities, cur_node.depth)

        # check to make sure we have a valid split, stop and make leaf if not
        if split_feature_index == -1:

            cur_node.probabilities = self.generate_probabilities(labels)
            cur_node.pred = np.argmax(cur_node.probabilities)
            cur_node.pred_prob = cur_node.probabilities[cur_node.pred]

            self.train_error += (
                (1 - cur_node.pred_prob) * len(labels) / self.total_samples
            )
            self.leaves += 1
            self.leaves_.append(cur_node)

        else:
            # update cur node
            cur_node.split_feature = cur_node.split_features[split_feature_index]
            cur_node.split_point = cur_node.split_points[split_feature_index]

            # update feature importances
            if cur_node.split_feature >= self.total_features:
                importances_feature = cur_node.split_feature - self.total_features
            else:
                importances_feature = cur_node.split_feature

            if not self.size_scale:
                self.feature_importances[importances_feature] += 1

            else:
                self.feature_importances[importances_feature] += len(features)

            # divide data accordingly for next node
            (
                left_features,
                left_labels,
                right_features,
                right_labels,
            ) = GenerousTree.split_data_2(
                features, labels, cur_node.split_feature, cur_node.split_point
            )

            # create left and right nodes using node constructor if we have made it this far
            cur_node.left = GenerousNode(parent=cur_node, depth=cur_node.depth + 1)
            cur_node.right = GenerousNode(parent=cur_node, depth=cur_node.depth + 1)

            # fit left and right nodes
            self.nodes += 2
            self.ovr_fit_inner(left_features, left_labels, cur_node=cur_node.left)
            self.ovr_fit_inner(right_features, right_labels, cur_node=cur_node.right)

    def nudge_split(self, features, labels, boundary_std):
        """
        train two nudge splitters working in opposite directions, then pick the one with best error

        """

        start_forward = np.median(features)
        print(start_forward)
        start_backward = start_forward
        feature_len = len(features)

        # print(features)
        # print(labels)
        # print(start_forward, start_backward)

        for i in range(feature_len * self.ovr_cycles):

            if features[i % feature_len] >= start_forward:

                if labels[i % feature_len] != 1:

                    start_forward += self.boundary_multiplier * boundary_std

            else:
                if labels[i % feature_len] != 0:
                    start_forward -= self.boundary_multiplier * boundary_std

            if features[i % feature_len] >= start_backward:

                if labels[i % feature_len] != 0:

                    start_backward += self.boundary_multiplier * boundary_std

            else:
                if labels[i % feature_len] != 1:
                    start_backward -= self.boundary_multiplier * boundary_std

        fwd_err, backward_err = GenerousTree.check_fwd_backwd(
            start_forward, start_backward, features, labels
        )

        if fwd_err < backward_err:
            return (fwd_err, start_forward)

        else:
            return (backward_err, start_backward)

    @staticmethod
    def check_fwd_backwd(start_fwd, start_back, features, labels):
        """
        get zero one error for fwd and backward split points
        """

        # first do fwd
        fwd_left = labels[np.where(features < start_fwd)]
        fwd_right = labels[np.where(features >= start_fwd)]

        # print(f"left: {len(fwd_left)}")
        # print(f"right: {len(fwd_right)}")

        try:
            fwd_score = (
                Counter(fwd_left).most_common(1)[0][1]
                + Counter(fwd_right).most_common(1)[0][1]
            ) / len(labels)
        except:
            fwd_score = sys.maxsize

        back_left = labels[np.where(features < start_back)]
        back_right = labels[np.where(features >= start_back)]

        try:
            back_score = (
                Counter(back_left).most_common(1)[0][1]
                + Counter(back_right).most_common(1)[0][1]
            ) / len(labels)
        except:
            back_score = sys.maxsize

        return fwd_score, back_score

    def get_best_n(self, depth):
        """
        get new int output of how many best n to consider, never less than one

        inputs:
                depth: int: depth of current node

        outputs:
                best_n: how many best n to consider
        """

        best_n = self.best_n_scale(depth, self.best_n)

        if best_n < 1:
            return 1
        else:
            return int(best_n)

    def pb_tree_clean(self, cur_node):
        """
        create pb_tree compatible with original features
        """

        if cur_node.split_feature is not None:

            if cur_node.split_feature >= self.total_features:

                cur_node.split_feature = cur_node.split_feature - self.total_features
                cur_node.split_point = -cur_node.split_point
                temp = cur_node.left
                cur_node.left = cur_node.right
                cur_node.right = temp

            self.pb_tree_clean(cur_node=cur_node.left)
            self.pb_tree_clean(cur_node=cur_node.right)

    def pb_ovr_refit(self, features, labels, pb_features=False):
        """ """
        if not pb_features:

            features_ = GenerousTree.pb_preprocess_features(features)

        else:
            features_ = features

        # refit inner will call itself recrusively
        self.populate_feature_stds(features_, self.root)

        self.pb_ovr_refit_inner(features_, labels)

        self.pb_tree_clean(cur_node=self.root)

    def pb_ovr_refit_inner(self, features_, labels):
        """ """
        feature_len = len(features_)
        # first check stop conditions
        for i in range(self.max_iter):

            # get the prediction
            leaf = self.fwd_ovr(features_[i % feature_len])
            loss = self.pb_loss(leaf, labels[i % feature_len])

            if loss > 0:
                # send backward to update split points
                self.ovr_backward(leaf, loss, features_[i % feature_len])

    def fwd_ovr(self, instance):
        """ """
        cur_node = self.root

        while cur_node.left:

            if instance[cur_node.split_feature] < cur_node.split_point:

                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        return cur_node

    def ovr_backward(self, leaf, loss, instance):
        """ """

        while leaf.parent:

            # move to parent node
            leaf = leaf.parent
            if instance[leaf.split_feature] < leaf.split_point:

                leaf.split_point -= (
                    loss
                    * self.boundary_multiplier
                    * leaf.feature_stds[leaf.split_feature]
                )

            else:
                leaf.split_point += (
                    loss
                    * self.boundary_multiplier
                    * leaf.feature_stds[leaf.split_feature]
                )

    def populate_feature_stds(self, features, cur_node):

        if cur_node.left:
            cur_node.feature_stds = np.zeros(features.shape[1])

            for i in range(features.shape[1]):
                cur_node.feature_stds[i] = mad(features[:, i])

            left_features = features[
                np.asarray(
                    features[:, cur_node.split_feature] < cur_node.split_point
                ).nonzero()
            ]
            right_features = features[
                np.asarray(
                    features[:, cur_node.split_feature] >= cur_node.split_point
                ).nonzero()
            ]

            self.populate_feature_stds(left_features, cur_node.left)
            self.populate_feature_stds(right_features, cur_node.right)

    def pb_refit(self, features, labels, cur_node, pb_features=False, refit_all=False):
        """
        tree
        """

        # #make sure all attributes up to date
        # if self.forest:
        #     self.align_attrs()

        if not pb_features:

            features_ = GenerousTree.pb_preprocess_features(features)

        else:
            features_ = features

        # refit inner will call itself recrusively
        self.pb_refit_inner(features_, labels, cur_node, refit_all=refit_all)

        self.pb_tree_clean(cur_node=self.root)

    def pb_refit_inner(self, features_, labels, cur_node, refit_all):
        """
        inner method for tree pb_refit
        """

        # first check stop conditions
        if (
            len(features_) < self.min_samples_split
            or self.check_node_impurity(labels) == 0
            or cur_node.depth == self.max_depth
        ):

            # generate prediction for leaf
            self.leaves += 1
            cur_node.probabilities = self.generate_probabilities(labels)
            cur_node.pred = np.argmax(cur_node.probabilities)
            cur_node.left = None
            cur_node.right = None
            cur_node.split_feature = None

        # see if this node has a left
        elif cur_node.left:

            # find optimal split point at this node
            # change this so that we find the best split across all features each time we do this...maybe except for last iteration
            if refit_all:
                # gather all impurities and split points by feature...sorted by impurity
                split_points = np.zeros(len(cur_node.split_points_))
                for i in range(len(cur_node.split_features_)):

                    _, split_point, good_split_flag = self.find_best_split_2(
                        features_[:, cur_node.split_features_[i]], labels
                    )

                    split_points[i] = split_point

                    cur_node.split_points_ = split_points

            # recompute to ensure good split flag
            _, split_point, good_split_flag = self.find_best_split_2(
                features_[:, cur_node.split_feature], labels
            )

            # check to see that we have a valid split
            if good_split_flag:

                # change the split point of current node only if split is valid
                if cur_node.split_point != split_point:
                    cur_node.split_point = split_point
                    self.split_points_changed += 1

                (
                    left_features,
                    left_labels,
                    right_features,
                    right_labels,
                ) = self.split_data_2(
                    features_, labels, cur_node.split_feature, cur_node.split_point
                )

                self.nodes += 2
                self.pb_refit(
                    left_features, left_labels, cur_node=cur_node.left, pb_features=True
                )
                self.pb_refit(
                    right_features,
                    right_labels,
                    cur_node=cur_node.right,
                    pb_features=True,
                )

            # this is now a leaf
            else:

                # generate prediction for leaf
                self.leaves += 1
                cur_node.probabilities = self.generate_probabilities(labels)
                cur_node.pred = np.argmax(cur_node.probabilities)
                cur_node.left = None
                cur_node.right = None
                cur_node.split_feature = None

        # this node was a leaf
        else:

            # gather all impurities and split points by feature...sorted by impurity
            # only want to use the original features here!!!!!!!!!

            (
                cur_node.split_features,
                impurities,
                cur_node.split_points,
            ) = self.split_feature_aggregate(
                features_[:, : int(features_.shape[1] / 2)], labels
            )

            # select the split feature by applying chosen probability transform
            (
                split_feature_index,
                cur_node.split_prob,
                cur_node.feature_probs,
            ) = self.split_feature_select(impurities, cur_node_depth=cur_node.depth)

            # catch case where there were no valid splits
            if split_feature_index == -1:

                # generate prediction for leaf
                self.leaves += 1
                cur_node.probabilities = self.generate_probabilities(labels)
                cur_node.pred = np.argmax(cur_node.probabilities)
                cur_node.left = None
                cur_node.right = None
                cur_node.split_feature = None

            # otherwise there is a valid split
            else:

                # change pred to None
                cur_node.pred = None

                # update cur node
                cur_node.split_feature = cur_node.split_features[split_feature_index]
                cur_node.split_point = cur_node.split_points[split_feature_index]

                # update feature importances

                if cur_node.split_feature >= self.total_features:
                    importances_feature = cur_node.split_feature - self.total_features
                else:
                    importances_feature = cur_node.split_feature

                if not self.size_scale:
                    self.feature_importances[importances_feature] += 1
                else:
                    self.feature_importances[importances_feature] += (
                        cur_node.parent.times_traversed / 2
                    )

                (
                    left_features,
                    left_labels,
                    right_features,
                    right_labels,
                ) = self.split_data_2(
                    features_, labels, cur_node.split_feature, cur_node.split_point
                )
                cur_node.left = GenerousNode(parent=cur_node, depth=cur_node.depth + 1)
                cur_node.right = GenerousNode(parent=cur_node, depth=cur_node.depth + 1)

                self.nodes += 2
                self.pb_refit_inner(
                    left_features,
                    left_labels,
                    cur_node=cur_node.left,
                    refit_all=refit_all,
                )
                self.pb_refit_inner(
                    right_features,
                    right_labels,
                    cur_node=cur_node.right,
                    refit_all=refit_all,
                )

    def predict(
        self,
        features,
        pb_features=False,
        return_node=False,
    ):
        """
        tree predict

        method to generate predictions from test data is original label format (uses pred_map)

        inputs:
                features: nxd array of feature values, same format as train features
                map_labels: bool: use self.label_map_pred to convert labels to original format
                pb_features: bool: whether features have been transformed to pb format
                return_node: bool: whether to return the leaf node instead of the prediction

        outputs:
                preds: nx1 array of predicted classes
                probs: nxj probability array from the leaf node used (j is num_possible classes)

        calls:
                generous_tree.traverse
        """

        # allocate output array for outputs that we need
        y_hat = np.empty(len(features), dtype=object)

        if pb_features:
            features_ = GenerousTree.pb_preprocess_features(features)

        else:
            features_ = features

        # pass over each row in features
        for i in range(len(features_)):

            # include probabilities if necessary
            leaf_node = GenerousTree.traverse(self.root, features_[i])

            if not return_node:

                y_hat[i] = self.label_map_pred[leaf_node.pred]

            else:
                y_hat[i] = leaf_node

        return y_hat

    @staticmethod
    def traverse(root, features):
        """
        given an array of feature values, traverse the generous tree until a leaf node is reached
        returning the majority feature

        inputs:
                root: node object at base of decision tree
                features: 1xd array of feature values to predict one label
                include_probs: bool: whether to return probabilities from leaf node where we end up

        returns:
                leaf node where we end up in tree

        calls:
                None
        """

        # check to see if we are in leaf
        while root.left:

            # check whether we go left or right if not in leaf
            if features[root.split_feature] < root.split_point:

                root = root.left

            else:
                root = root.right

        return root

    def fit(self, features, labels):
        """
        tree fit

        """
        self.total_samples = len(labels)
        self.total_features = features.shape[1]
        self.feature_importances = np.zeros(self.total_features)

        # check to see if we have generated slots and label map
        if self.label_map_train is None:

            (
                self.slots,
                self.label_map_train,
                self.label_map_pred,
            ) = GenerousTree.preprocess_labels(labels)

            self.labels_processed = True
            self.total_samples = len(labels)

        if type(self.max_features) != int and self.max_features is not None:

            self.get_max_features(features)

        # only create the root on the first split, otherwise we will have passed a non-null node
        self.nodes += 1
        self.root = GenerousNode(parent=None, depth=1)

        # call fit_inner to grow tree
        self.fit_inner(features, labels, cur_node=self.root)

    ######################PB Methods###############################################
    def pb_preprocess_tree(self, cur_node=None):
        """
        traverse nodes in tree and set feature_probs_ for each node
        """
        if cur_node is None:
            cur_node = self.root

        # this is not a leaf
        if cur_node.left:

            # first create feature_probs_ array of correct length and normalize
            cur_node.feature_probs_ = np.append(
                cur_node.feature_probs, cur_node.feature_probs
            )
            cur_node.feature_probs_ = cur_node.feature_probs_ / sum(
                cur_node.feature_probs_
            )

            # then add the properly scaled overlay
            cur_node.feature_probs_ = cur_node.feature_probs_ + (
                self.pb_uniform_overlay * np.ones(len(cur_node.feature_probs_))
            ) / len(cur_node.feature_probs_)
            cur_node.feature_probs_ = cur_node.feature_probs_ / sum(
                cur_node.feature_probs_
            )

            cur_node.split_features_ = np.append(
                cur_node.split_features, self.total_features + cur_node.split_features
            )
            cur_node.split_points_ = np.append(
                cur_node.split_points, -1 * cur_node.split_points
            )

            # try to call on child nodes
            self.pb_preprocess_tree(cur_node=cur_node.left)
            self.pb_preprocess_tree(cur_node=cur_node.right)

    @staticmethod
    def pb_preprocess_features(features):
        """
        creates negation of feature set and concatenates it onto original
        """

        return np.concatenate((features, -features), axis=1)

    def pb_train(self, features, labels, warm_start=False, pb_features=False):
        """
        trains tree using psuedo backprop

        ***need to raise error if dimension does not match self.total features*****

        """
        if not pb_features:
            features_ = GenerousTree.pb_preprocess_features(features)

        else:
            features_ = features

        # prepare tree nodes for pb (fills in feature_probs_ for each node)
        if not warm_start:

            if self.max_features is not None:
                self.max_features_ = self.max_features * 2

            else:
                self.max_features_ = features_.shape[1]

            self.pb_preprocess_tree(self.root)

        self.pb_score = self.pb_train_inner(features_, labels)

        # create new feature importances
        self.feature_importances = np.zeros(self.total_features)
        self.make_pb_tree()

    def pb_train_inner(self, features, labels):
        """ """
        # generate random indices up to max_iter size
        inds = np.random.randint(0, len(features), size=int(self.max_iter))

        # pass over rows of data, send them through tree probabilisitically
        j = 0
        k = 0
        for i in inds:

            # grab leaf from probabilistic matriculation through tree
            leaf = self.forward(features[i])

            # calculate loss
            loss = self.pb_loss(leaf, self.label_map_train[labels[i]])
            k += loss

            # check to see if we need to do backwards pass
            if loss > 0:

                num_perfect = 0
                self.backward(leaf.parent, loss)

            j += 1
            if j % self.pb_block_size == 0:

                if k / self.pb_block_size > self.early_stop_thresh:

                    self.early_stop = True
                    return k / self.pb_block_size

                if j < self.max_iter:
                    k = 0

        return k / self.pb_block_size

    def forward(self, instance):

        cur_node = self.root
        cur_node.times_traversed += 1

        while cur_node.left:

            # choose index of feature to split on
            cur_node.active_feature_index = np.random.choice(
                self.max_features_, p=cur_node.feature_probs_
            )

            # check to see if we are greater than randomly chosen split feature
            if (
                instance[cur_node.split_features_[cur_node.active_feature_index]]
                < cur_node.split_points_[cur_node.active_feature_index]
            ):

                cur_node = cur_node.left
                cur_node.times_traversed += 1

            else:
                cur_node = cur_node.right
                cur_node.times_traversed += 1

        # add this here in case we end up splitting this leaf in pb refit
        cur_node.times_traversed += 1

        # return leaf when we can no child node
        return cur_node

    def backward(self, cur_node, loss):

        # will end once cur_node reaches the root as cur_node.parent will be None
        while cur_node:

            # update the probability of the feature index that did not work
            if self.depth_scale:
                cur_node.feature_probs_[cur_node.active_feature_index] -= (
                    loss * self.lambda_ * cur_node.depth
                )

            else:
                cur_node.feature_probs_[cur_node.active_feature_index] -= (
                    loss * self.lambda_
                )

            # check to make sure no negative probabilities
            if cur_node.feature_probs_[cur_node.active_feature_index] < 0:
                cur_node.feature_probs_[cur_node.active_feature_index] = 0

            # renormalize
            cur_node.feature_probs_ = cur_node.feature_probs_ / sum(
                cur_node.feature_probs_
            )
            cur_node = cur_node.parent

    ##TRAINING METHODS########################################################################################
    def fit_inner(self, features, labels, cur_node):
        """

        fits the decision tree recursively until some stop condition is met

        features is nXd array
        labels is nx1 array
        """

        # update cur_node
        cur_node.impurity = self.check_node_impurity(labels)
        cur_node.num_instances = len(labels)

        # check stop conditions for recursion, generate probabilities if so
        if (
            len(features) < self.min_samples_split
            or cur_node.impurity == 0
            or cur_node.depth == self.max_depth
        ):

            cur_node.probabilities = self.generate_probabilities(labels)
            cur_node.pred = np.argmax(cur_node.probabilities)
            cur_node.pred_prob = max(cur_node.probabilities)

            self.train_error += (
                (1 - cur_node.pred_prob) * len(labels) / self.total_samples
            )
            self.leaves += 1
            self.leaves_.append(cur_node)

            return

        # gather all impurities and split points by feature...sorted by impurity
        (
            cur_node.split_features,
            impurities,
            cur_node.split_points,
        ) = self.split_feature_aggregate(features, labels)

        # select the split feature by applying chosen probability transform
        (
            split_feature_index,
            cur_node.split_prob,
            cur_node.feature_probs,
        ) = self.split_feature_select(impurities, cur_node.depth)

        # check to make sure we have a valid split, stop and make leaf if not
        if split_feature_index == -1:

            cur_node.probabilities = self.generate_probabilities(labels)
            cur_node.pred = np.argmax(cur_node.probabilities)
            cur_node.pred_prob = cur_node.probabilities[cur_node.pred]

            self.train_error += (
                (1 - cur_node.pred_prob) * len(labels) / self.total_samples
            )
            self.leaves += 1
            self.leaves_.append(cur_node)

        else:
            # update cur node
            cur_node.split_feature = cur_node.split_features[split_feature_index]
            cur_node.split_point = cur_node.split_points[split_feature_index]

            # update feature importances
            if cur_node.split_feature >= self.total_features:
                importances_feature = cur_node.split_feature - self.total_features
            else:
                importances_feature = cur_node.split_feature

            if not self.size_scale:
                self.feature_importances[importances_feature] += 1

            else:
                self.feature_importances[importances_feature] += len(features)

            # divide data accordingly for next node
            (
                left_features,
                left_labels,
                right_features,
                right_labels,
            ) = GenerousTree.split_data_2(
                features, labels, cur_node.split_feature, cur_node.split_point
            )

            # create left and right nodes using node constructor if we have made it this far
            cur_node.left = GenerousNode(parent=cur_node, depth=cur_node.depth + 1)
            cur_node.right = GenerousNode(parent=cur_node, depth=cur_node.depth + 1)

            # fit left and right nodes
            self.nodes += 2
            self.fit_inner(left_features, left_labels, cur_node=cur_node.left)
            self.fit_inner(right_features, right_labels, cur_node=cur_node.right)

    def transform_impurities(self, imps):

        return self.transform_method(-self.scaling * imps)

    def get_max_features(self, features):

        self.max_features = int(self.max_features(features.shape[1]))

    def split_feature_select(self, impurities, cur_node_depth):
        """
        applies chosen probability transform method to impurities, then selects one index at random according to
        the probabilities of each index after transform. steps below

        1)apply method (function) to array containing raw impurities
        2)normalize raw impurities
        3)choose index at random according to normalized probabilities

        inputs:
                impurities: dx1 array: raw impurities given the scoring method chosen
                cur_node_depth: int: depth of current node

        returns:
                (feature, feature_prob, transformed_impurities)
                feature: int: index of feature chosen
                feature_prob: transformed, normalized probability of having chosen this feature
                transformed_impurities: array of transformed, normalized impurities

        calls:
                transform_impurities
        """

        # apply transformation
        transformed_impurities = self.transform_impurities(impurities)

        # catch case that there were no valid splits before we hit div 0 error
        if sum(transformed_impurities) == 0:

            return (-1, 0, 0)

        # sum and normalize
        transformed_impurities = transformed_impurities / sum(transformed_impurities)

        if self.best_n_scale is not None:
            best_n = self.get_best_n(cur_node_depth)

        else:
            best_n = self.best_n
        # choose feature
        choice = np.random.choice(
            range(len(impurities))[:best_n],
            p=transformed_impurities[:best_n] / np.sum(transformed_impurities[:best_n]),
        )

        return (choice, transformed_impurities[choice], transformed_impurities)

    @staticmethod
    def preprocess_features(features):
        """
        ensure that features are numeric and that all columns contain at least 2 unique values

        inputs:
                features: nxd array of all feature values

        returns:
                problem_columns: 1d array of column indices that don't pass check

        calls:
                None
        """
        problem_columns = list()

        for i in range(features.shape[1]):

            unique_values = set(features[:, i])

            if len(unique_values) < 2:
                problem_columns.append(i)
                next

            for j in unique_values:

                if type(j) != int or type(j) != float:

                    problem_columns.append(i)

        return problem_columns

    @staticmethod
    def preprocess_labels(labels):
        """
        convert labels to numeric 0...d-1 array labels if necessary and generate slots

        inputs:
                labels: unconverted labels (could be numeric or not)

        returns:
                (slots, labels_map)
                slots: int: number of unique labels in set
                label_map_train: dict: maps observed label to 0...d-1 for d slots for probability arrays in leaf nodes
                label_map_pred: dict: maps 0...d-1 to original label

        calls:
                None
        """
        # convert label array to set and initialize dictionary to serve as map
        slots = set(labels)
        label_map_train = dict()
        label_map_pred = dict()

        # order does not matter, just unique map 0...d-1
        counter = 0
        for i in slots:

            label_map_train[i] = counter
            label_map_pred[counter] = i
            counter += 1

        return (len(slots), label_map_train, label_map_pred)

    def generate_probabilities(self, labels):
        """
        generate array of probabilties of different labels for this leaf

        inputs:
                labels: nx1 array: labels associated with this node
                label_map: dict: maps label value to position in array
                slots: int: the number of possible

        returns:
                probabilities: dx1 array: observed proportion of all d possible labels in labels

        calls:
                None
        """

        probs = np.zeros(self.slots)

        for i in labels:

            probs[self.label_map_train[i]] += 1

        return probs / sum(probs)

    @staticmethod
    def split_data(features, labels, split_feature, split_point):
        """
        splits feature array and labels based on a split point for a given feature

        inputs:
                features: nxd array: Values for all features
                labels: nx1 array: labels with order corresponding to features
                split_feature: int: index of feature to split on
                split_point: numeric: x < split_point -> left node features and labels

        returns:
                (left_features, left_labels, right_features, right_labels)

                all arrays, nxd for features and nx1 for labels that belong in left and right nodes

        calls:
                None
        """

        # initialize two lists to hold the indices for left and right (will be the same for features and labels)
        left_inds = list()
        right_inds = list()

        # check each "row" in features in the correct column, and update left or right inds
        for i in range(features.shape[0]):

            if features[i][split_feature] < split_point:

                left_inds.append(i)

            else:

                right_inds.append(i)

        return (
            features[left_inds],
            labels[left_inds],
            features[right_inds],
            labels[right_inds],
        )

    @staticmethod
    def split_data_2(features, labels, split_feature, split_point):
        """
        splits feature array and labels based on a split point for a given feature

        inputs:
                features: nxd array: Values for all features
                labels: nx1 array: labels with order corresponding to features
                split_feature: int: index of feature to split on
                split_point: numeric: x < split_point -> left node features and labels

        returns:
                (left_features, left_labels, right_features, right_labels)

                all arrays, nxd for features and nx1 for labels that belong in left and right nodes

        calls:
                None
        """

        # initialize two lists to hold the indices for left and right (will be the same for features and labels)
        left_inds = np.asarray(features[:, split_feature] < split_point).nonzero()
        right_inds = np.asarray(features[:, split_feature] >= split_point).nonzero()

        return (
            features[left_inds],
            labels[left_inds],
            features[right_inds],
            labels[right_inds],
        )

    def split_feature_aggregate(self, features, labels):
        """
        randomly selects feature on which to split according to relative impurities resulting from feature splits
        best split is determined for each feature and then relative impurities of features are weighed according
        to the BEST SPLIT for each of those features.

        ***RETURNS SORTED BY IMPURITY****

        inputs:
                features: nxd array of train data
                labels: nx1 array of labels
                refit: bool - if true use max_features_

        returns:
                (feature_subset, impurities, split_points)
                feature_subset: dx1 array: indices of features examined
                impurities: dx1 array: raw impurities associated with each feature
                split_points: dx1 array: split points associate with each feature

        calls:
                GenerousTree.find_best_split_2
        """
        # initialize array of impurities and hold onto split point, features, and labels

        if self.max_features is None:
            # consider all features
            feature_subset = np.arange(features.shape[1])
            impurities = np.zeros(features.shape[1])
            split_points = np.zeros(features.shape[1])

        else:
            # pick random subset of features
            feature_subset = np.random.choice(
                features.shape[1], size=self.max_features, replace=False
            )

            impurities = np.zeros(self.max_features)
            split_points = np.zeros(self.max_features)

        # loop over feature_subset
        for i in range(len(feature_subset)):

            impurity_score, split_point, good_split_flag = self.find_best_split_2(
                features[:, feature_subset[i]], labels
            )

            if good_split_flag:

                impurities[i] = impurity_score
                split_points[i] = split_point

            else:

                impurities[i] = sys.maxsize
                split_points[i] = split_point

        sort_order = np.argsort(impurities)
        feature_subset = feature_subset[sort_order]

        impurities = impurities[sort_order]
        split_points = split_points[sort_order]

        return (feature_subset, impurities, split_points)

    def find_best_split_2(self, feature_values, labels):
        """
        Linear time find best split

        consider using collections-counter to speed up
        """
        # sort features and values before beginning
        sort_order = np.argsort(feature_values)
        ordered_labels = labels[sort_order]
        ordered_features = feature_values[sort_order]

        # create condensed arrays to speed up impurity computation
        condensed_right = np.zeros(self.slots)
        condensed_left = np.zeros(self.slots)

        # first pass over labels to create condensed_right
        for i in ordered_labels:

            condensed_right[self.label_map_train[i]] += 1

        # initialize best split impurity measure and split point tracker
        impurity_score = self.check_node_impurity_frequencies(condensed_right)
        split_point = ordered_features[0]
        good_split_flag = False
        cur = ordered_features[0]

        # check to see if features are uniform...this will be picked up in fit
        if cur == ordered_features[-1]:

            return (impurity_score, split_point, good_split_flag)

        for i in range(len(ordered_features)):

            # check to see if new value is actaully greater than the previous one
            # if yes, we need to calculate impurity, keeping track of lowest score
            if ordered_features[i] > cur:

                # check contributions to impurity from left and right before we increment i
                left_impurity = self.check_node_impurity_frequencies(condensed_left) * i
                right_impurity = self.check_node_impurity_frequencies(
                    condensed_right
                ) * (ordered_features.size - i)

                # compare impurity for this split to minimum value we have observed so far
                split_impurity = (
                    left_impurity + right_impurity
                ) / ordered_features.size

                # update best split impurity and split point
                if split_impurity < impurity_score:

                    # we have found a valid split
                    good_split_flag = True

                    # update impurity and split point
                    impurity_score = split_impurity
                    split_point = (ordered_features[i] + cur) / 2

                    # check for perfect score
                    if impurity_score == 0:

                        return (impurity_score, split_point, good_split_flag)

                # update cur before I forget
                cur = ordered_features[i]

            # move the label from right to left
            condensed_right[self.label_map_train[ordered_labels[i]]] -= 1
            condensed_left[self.label_map_train[ordered_labels[i]]] += 1

        return (impurity_score, split_point, good_split_flag)

    def find_best_split(self, feature_values, labels):
        """
        inputs:
                feature_values: nx1 array of feature VALUES
                labels: nx1 array of labels
                cutoff_value: (numeric) less than cutoff values

        returns:
                (impurity_score, split_point)
                cutoff_point: if feature_value < cutoff point -> left node
                impurity_score: self explanatory

        calls:
                GenerousTree.check_node_impurity
        """

        # create copies according to order of sorted feature values
        sort_order = np.argsort(feature_values)
        ordered_labels = labels[sort_order]

        # initialize best split impurity measure and split point tracker
        impurity_score = sys.maxsize
        split_point = 0

        # begin from smallest value
        cur = feature_values[sort_order[0]]

        # check to make sure features are not uniform in this node

        for i in range(len(sort_order)):

            # check to see if new value is actaully greater than the previous one
            # if yes, we need to calculate impurity, keeping track of lowest score
            if feature_values[sort_order[i]] > cur:

                # update cur before I forget
                cur = feature_values[sort_order[i]]

                # check contributions to impurity from left and right before we increment i
                left_impurity = self.check_node_impurity(ordered_labels[:i]) * i
                right_impurity = self.check_node_impurity(ordered_labels[i:]) * (
                    len(sort_order) - i
                )

                # compare impurity for this split to minimum value we have observed so far
                split_impurity = (left_impurity + right_impurity) / len(sort_order)

                # update best split impurity and split point
                if split_impurity < impurity_score:

                    impurity_score = split_impurity
                    split_point = feature_values[sort_order[i]]

        return (impurity_score, split_point)

    def check_node_impurity(self, labels):
        """
        check the impurity for 1! node
        should this be in the node class?

        inputs:
                labels: array: labels for this node
                criterion: str: "gini" or "entropy"

        returns:
                impurity: float: raw impurity score

        calls:
                None
        """
        impurity = 0
        if self.criterion == "gini":

            instances_dict = Counter(labels)

            for i in instances_dict.values():

                p = i / sum(instances_dict.values())
                impurity += p * (1 - p)

        elif self.criterion == "entropy":

            value, counts = np.unique(labels, return_counts=True)
            impurity = entropy(counts)

        return impurity

    def check_node_impurity_frequencies(self, label_freqs):
        """
        check the impurity for 1! node
        should this be in the node class?

        inputs:
                labels: array: label frequencies for this node
                criterion: str: "gini" or "entropy"

        returns:
                impurity: float: raw impurity score

        calls:
                None
        """
        impurity = 0

        if self.criterion == "gini":

            return (label_freqs / np.sum(label_freqs)) @ (
                1 - (label_freqs / np.sum(label_freqs))
            )

        elif self.criterion == "entropy":

            impurity = entropy(label_freqs)

        return impurity

    def make_pb_tree(self, cur_node=None):
        """
        create pb trees after doing pb training

        root: generous node- root of decision tree after pb training
        refit: bool: whether to reoptimize breakpoints by feature to maximize info gain

        """

        if cur_node is None:
            cur_node = self.root

        if cur_node.left:

            index = np.argmax(cur_node.feature_probs_)
            split_feature = cur_node.split_features_[index]

            if split_feature >= self.total_features:
                importances_feature = split_feature - self.total_features
            else:
                importances_feature = split_feature

            if not self.size_scale:
                self.feature_importances[importances_feature] += 1
            else:
                self.feature_importances[
                    importances_feature
                ] += cur_node.times_traversed

            if split_feature != cur_node.split_feature:
                self.split_features_changed += 1
                cur_node.split_feature = split_feature
                cur_node.split_point = cur_node.split_points_[index]

            self.make_pb_tree(cur_node.left)
            self.make_pb_tree(cur_node.right)

    ###############################################################################################################
