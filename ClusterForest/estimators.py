from .GenerousTree import GenerousTree
from . import VoteMethods
from . import Losses
import numpy as np
import copy


class Sequential:
    """
    Class for stringing together clustering/Generous forest methods
    """

    def __init__(self, clf, calls):
        self.clf = clf
        self.calls = calls

    def run(self):
        """
        run calls in order on self.clf
        """

        for i in self.calls:

            if type(i) == str:
                call = getattr(self.clf, i)
                call()

            elif type(i) == dict:

                for j in i:
                    setattr(self.clf, j, i[j])


class GenerousForest:
    """
    generous forest classifier

    composition using GenerousNode, GenerousTree
    """

    def __init__(
        self,
        criterion="gini",
        n_estimators=100,
        transform_method=np.exp,
        best_n=None,
        scaling=1,
        vote_method=VoteMethods.majority_vote,
        max_features=None,
        bootstrap=True,
        maintain_bootstrap=False,
        early_stop_thresh=0.95,
        max_iter=2e4,
        lambda_=0.005,
        depth_scale=False,
        slots=None,
        pb_block_size=1e4,
        pb_loss=Losses.zero_one_loss,
        size_scale=False,
        pb_uniform_overlay=0,
        em_iterations=1,
        baseline_iterations=1,
        eval_loss=Losses.zero_one_loss_eval,
        cluster_iterations=0,
        max_depth=None,
        best_n_scale=None,
        ovr_baseline=False,
        ovr_cycles=5,
        boundary_multiplier=1,
    ):

        self.n_estimators = n_estimators
        self.transform_method = transform_method
        self.best_n = best_n
        self.scaling = scaling
        self.criterion = criterion
        self.vote_method = vote_method
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.maintain_bootstrap = maintain_bootstrap
        self.max_iter = int(max_iter)
        self.early_stop_thresh = early_stop_thresh
        self.lambda_ = lambda_
        self.depth_scale = depth_scale
        self.slots = slots
        self.pb_block_size = pb_block_size
        self.pb_loss = pb_loss
        self.size_scale = size_scale
        self.pb_uniform_overlay = pb_uniform_overlay
        self.em_iterations = em_iterations
        self.baseline_iterations = baseline_iterations
        self.eval_loss = eval_loss
        self.cluster_iterations = cluster_iterations
        self.max_depth = max_depth
        self.best_n_scale = best_n_scale
        self.ovr_baseline = ovr_baseline
        self.ovr_cycles = ovr_cycles
        self.boundary_multiplier = boundary_multiplier

        self.ave_tree_train_error = 0
        self.ave_leaves_per_tree = 0
        self.ave_nodes_per_tree = 0
        self.trees = np.empty(self.n_estimators, dtype=object)
        self.label_map_pred = None
        self.label_map_train = None
        self.early_stops = 0
        self.ave_pb_score = 0
        self.ave_split_points_changed = 0
        self.ave_split_features_changed = 0
        self.feature_importances = None
        self.error_trajectory = None
        self.cluster_model = list()

    def create_tree(self):

        (
            method,
            best_n,
            scaling,
            criterion,
            max_features,
        ) = self.get_hyperparams()

        # create tree
        tree = GenerousTree(
            criterion=criterion,
            transform_method=method,
            best_n=best_n,
            scaling=scaling,
            label_map_pred=self.label_map_pred,
            label_map_train=self.label_map_train,
            slots=self.slots,
            max_features=max_features,
            max_iter=self.max_iter,
            early_stop_thresh=self.early_stop_thresh,
            lambda_=self.lambda_,
            depth_scale=self.depth_scale,
            pb_block_size=self.pb_block_size,
            pb_loss=self.pb_loss,
            size_scale=self.size_scale,
            pb_uniform_overlay=self.pb_uniform_overlay,
            eval_loss=self.eval_loss,
            max_depth=self.max_depth,
            best_n_scale=self.best_n_scale,
            boundary_multiplier=self.boundary_multiplier,
            ovr_cycles=self.ovr_cycles,
        )

        return tree

    def thin_forest(self, best_n=None, pb_thresh=None):
        """
        remove trees that do not meet specification from forest(after pb_train only)
        """

        temp_trees = list()

        if pb_thresh is None:
            pb_thresh = 0

        for i in self.trees:

            if i.pb_score >= pb_thresh:
                temp_trees.append(i)

        temp_trees.sort(key=lambda x: x.pb_score, reverse=True)

        self.trees = np.array(temp_trees)[:best_n]

    def get_params(self):

        params = vars(self)

        return params

    @staticmethod
    def impose_bootstrap(bootstrap, features, labels):
        """
        samples data with replacement. sampled size varies according to self.bootstrap

        inputs:
                bootstrap: bool or fraction to determine size of bootstrap
                features: nxd array
                labels: nx1 array

        returns:
                (features_, labels_, indices) - bootstrap sample in new object, indices used
        """

        if bootstrap == True:

            bootstrap_ = 1
        else:
            bootstrap_ = bootstrap

        indices = np.random.choice(
            features.shape[0], size=int(bootstrap_ * (features.shape[0])), replace=True
        )

        return (features[indices], labels[indices], indices)

    def preprocess_labels(self, labels):
        """
        generate_slots

        inputs:
                labels: unconverted labels (could be numeric or not)

        returns:
                (slots, labels_map)
                slots: int: number of unique labels in set
                label_map_train: dict: maps observed label to 0...d-1 for d slots for probability arrays in leaf nodes
                label_map_pred: dict: maps 0...d-1 to original label
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

        self.slots = len(slots)
        self.label_map_train = label_map_train
        self.label_map_pred = label_map_pred

    def organize_hyperparams(self):
        """
        converts hyperparameters to tuple type to fit with hyperparameter sampling code (see get_hyperparameters)
        """

        for i in [
            "transform_method",
            "best_n",
            "scaling",
            "criterion",
            "max_features",
        ]:

            if type(getattr(self, i)) != tuple:

                setattr(self, i, ([getattr(self, i)], [1]))

    def get_hyperparams(self):
        """
        samples hyperparameters according to specified probability distribution in self.hyperparameter

        returns:
                out - tuple containing all hyperparameter values for this tree
        """

        out = list()
        for i in [
            self.transform_method,
            self.best_n,
            self.scaling,
            self.criterion,
            self.max_features,
        ]:

            out.append(np.random.choice(i[0], p=i[1]))

        return tuple(out)

    def fit_baseline(self, features, labels):
        """
        fit all trees s.t. oob error is minimized
        """

        self.total_features = features.shape[1]
        # only need to do these once to begin
        if self.label_map_pred is None:

            self.preprocess_labels(labels)
            self.organize_hyperparams()
            self.feature_importances = np.zeros(self.total_features)

        # keep same bootstrap features for each tree
        for tree_num in range(self.n_estimators):

            features_, labels_, indices = GenerousForest.impose_bootstrap(
                self.bootstrap, features, labels
            )

            # create mask for grabbing OOB features
            mask = np.ones(features.shape[0], dtype=bool)
            mask[indices] = False
            eval_features = features[mask]
            eval_labels = labels[mask]

            self.total_features = features.shape[1]

            tree = self.create_tree()

            # train tree and grab original prediction loss on OOB features
            tree.fit(features_, labels_)
            train_error = self.eval_loss(tree.predict(eval_features), eval_labels)

            # copy original tree
            temp = copy.deepcopy(tree)

            # do EM iterations
            for i in range(self.baseline_iterations):

                # fit copy of forest model for one EM step
                temp.fit(features_, labels_)

                # grab updated error for copy model
                train_error_temp = self.eval_loss(
                    temp.predict(eval_features), eval_labels
                )

                # see if error has improved, update tree if so
                if train_error_temp < train_error:

                    tree = copy.deepcopy(temp)
                    train_error = train_error_temp

                # make sure temp is up to date
                temp = copy.deepcopy(tree)

            self.trees[tree_num] = tree

    def em_fit_ovr(
        self, features, labels, eval_features=None, eval_labels=None, test_mode=False
    ):
        """
        em algorithm forest fit
        inputs:
                features: nxd array of features to train model on
                labels: nx1 array of labels to train model on
                eval_feautres: nxd array of features to predict on
                eval_labels: nx1 array of labels to evaluate model accuracy on
                maintain_bootstrap: bool - if true. bootstrap first, then em_train model with same resampled feature subset
                                           if False: follow bootstrap protocol defined by self.bootstrap
                                           Useful for keeping tree correlation low?
        """

        # train and retrain all trees on same set of bootstrapped features
        if self.maintain_bootstrap:

            self.total_features = features.shape[1]
            # only need to do these once to begin
            if self.label_map_pred is None:

                self.preprocess_labels(labels)
                self.organize_hyperparams()
                self.feature_importances = np.zeros(self.total_features)

            # keep same bootstrap features for each tree
            for tree_num in range(self.n_estimators):

                if self.bootstrap:

                    features_, labels_, indices = GenerousForest.impose_bootstrap(
                        self.bootstrap, features, labels
                    )

                    # create mask for grabbing OOB features
                    mask = np.ones(features.shape[0], dtype=bool)
                    mask[indices] = False
                    eval_features = features[mask]
                    eval_labels = labels[mask]

                else:
                    cut_point = int(len(features) * 0.8)

                    features_ = features[:cut_point]
                    eval_features = features[cut_point:]
                    labels_ = labels[:cut_point]
                    eval_labels = labels[cut_point:]

                # create tree
                tree = self.create_tree()

                # if in test mode we need to track errors over iterations
                if test_mode:
                    error_trajectory = list()

                # train tree and grab original prediction loss on OOB features
                if self.ovr_baseline:
                    tree.ovr_fit(features_, labels_)

                else:
                    tree.fit(features_, labels_)

                train_error = self.eval_loss(tree.predict(eval_features), eval_labels)

                if test_mode:
                    error_trajectory.append(train_error)

                # do baseline_iterations
                for _ in range(self.baseline_iterations - 1):

                    # recreate original tree
                    temp = self.create_tree()

                    # fit copy of forest model for one EM step
                    if self.ovr_baseline:
                        temp.ovr_fit(features_, labels_)

                    else:
                        temp.fit(features_, labels_)

                    # grab updated error for copy model
                    train_error_temp = self.eval_loss(
                        temp.predict(eval_features), eval_labels
                    )

                    # see if error has improved, update tree if so
                    if train_error_temp < train_error:

                        tree = copy.deepcopy(temp)
                        train_error = train_error_temp

                    if test_mode:
                        error_trajectory.append(
                            self.eval_loss(tree.predict(eval_features), eval_labels)
                        )

                # self.trees[tree_num] = tree
                train_error = self.eval_loss(tree.predict(eval_features), eval_labels)
                temp = copy.deepcopy(tree)

                tree.populate_feature_stds(features_, tree.root)

                # do EM iterations
                for _ in range(self.em_iterations):

                    # fit copy of forest model for one EM step
                    temp.pb_train(features_, labels_)
                    temp.pb_ovr_refit(features_, labels_)

                    # grab updated error for copy model
                    train_error_temp = self.eval_loss(
                        temp.predict(eval_features), eval_labels
                    )

                    # see if error has improved, update tree if so
                    if train_error_temp < train_error:

                        tree = copy.deepcopy(temp)
                        train_error = train_error_temp

                        if test_mode:
                            error_trajectory.append(
                                self.eval_loss(tree.predict(eval_features), eval_labels)
                            )

                    else:
                        if test_mode:
                            error_trajectory.append(
                                self.eval_loss(tree.predict(eval_features), eval_labels)
                            )

                    # make sure temp is up to date
                    temp = copy.deepcopy(tree)

                # update metrics
                self.feature_importances += temp.feature_importances
                self.ave_split_points_changed += (
                    tree.split_points_changed / self.n_estimators
                )
                self.ave_split_features_changed += (
                    tree.split_features_changed / self.n_estimators
                )
                if tree.early_stop:
                    self.early_stops += 1

                # finally, append tree to self.trees
                self.trees[tree_num] = tree

                if test_mode:
                    error_trajectory.append(
                        self.eval_loss(tree.predict(eval_features), eval_labels)
                    )

                    if self.error_trajectory is None:
                        self.error_trajectory = np.zeros(len(error_trajectory))

                    self.error_trajectory += (
                        np.array(error_trajectory) / self.n_estimators
                    )

        # otherwise, run fit, pb_fit, and pb_refit at the forest level
        else:
            features_ = features
            labels_ = labels

            if not eval_labels:
                eval_labels = labels_

            if not eval_features:
                eval_features = features_

            # fit initial model
            self.fit(features_, labels_)

            temp = copy.deepcopy(self)
            train_error = self.eval_loss(self.predict(eval_features), eval_labels)

            # do EM iterations
            for i in range(self.em_iterations - 1):

                # fit copy of forest model for one EM step
                temp.pb_fit(features_, labels_)
                temp.pb_refit(features_, labels_, refit_all=True)

                # grab updated error for copy model
                train_error_temp = temp.eval_loss(
                    temp.predict(eval_features), eval_labels
                )

                # see if error has improved, update self if so
                if train_error_temp < train_error:

                    self.trees = copy.deepcopy(temp.trees)
                    self.feature_importances = copy.deepcopy(temp.feature_importances)
                    self.early_stops = copy.deepcopy(temp.early_stops)
                    self.ave_split_points_changed = copy.deepcopy(
                        self.ave_split_points_changed
                    )
                    self.ave_split_features_changed = copy.deepcopy(
                        self.ave_split_features_changed
                    )

                    train_error = train_error_temp

                # reset temp to be a copy of original again
                temp = copy.deepcopy(self)

            # last iteration there is no need to refit all split points, and no need to recopy self
            temp.pb_fit(features_, labels_)
            temp.pb_refit(features_, labels_, refit_all=False)

            # grab updated error for copy model
            train_error_temp = self.eval_loss(temp.predict(eval_features), eval_labels)

            # see if error has improved, update self if so
            if train_error_temp < train_error:

                self.trees = copy.deepcopy(temp.trees)
                self.feature_importances = copy.deepcopy(temp.feature_importances)
                self.early_stops = copy.deepcopy(temp.early_stops)
                self.ave_split_points_changed = copy.deepcopy(
                    self.ave_split_points_changed
                )
                self.ave_split_features_changed = copy.deepcopy(
                    self.ave_split_features_changed
                )

    def em_fit(
        self, features, labels, eval_features=None, eval_labels=None, test_mode=False
    ):
        """
        em algorithm forest fit
        inputs:
                features: nxd array of features to train model on
                labels: nx1 array of labels to train model on
                eval_feautres: nxd array of features to predict on
                eval_labels: nx1 array of labels to evaluate model accuracy on
                maintain_bootstrap: bool - if true. bootstrap first, then em_train model with same resampled feature subset
                                           if False: follow bootstrap protocol defined by self.bootstrap
                                           Useful for keeping tree correlation low?
        """

        # train and retrain all trees on same set of bootstrapped features
        if self.maintain_bootstrap:

            self.total_features = features.shape[1]
            # only need to do these once to begin
            if self.label_map_pred is None:

                self.preprocess_labels(labels)
                self.organize_hyperparams()
                self.feature_importances = np.zeros(self.total_features)

            # keep same bootstrap features for each tree
            for tree_num in range(self.n_estimators):

                if self.bootstrap:

                    features_, labels_, indices = GenerousForest.impose_bootstrap(
                        self.bootstrap, features, labels
                    )

                    # create mask for grabbing OOB features
                    mask = np.ones(features.shape[0], dtype=bool)
                    mask[indices] = False
                    eval_features = features[mask]
                    eval_labels = labels[mask]

                else:
                    cut_point = int(len(features) * 0.8)

                    features_ = features[:cut_point]
                    eval_features = features[cut_point:]
                    labels_ = labels[:cut_point]
                    eval_labels = labels[cut_point:]

                # create tree
                tree = self.create_tree()

                # if in test mode we need to track errors over iterations
                if test_mode:
                    error_trajectory = list()

                # train tree and grab original prediction loss on OOB features
                tree.fit(features_, labels_)
                train_error = self.eval_loss(tree.predict(eval_features), eval_labels)

                if test_mode:
                    error_trajectory.append(train_error)

                # do baseline_iterations
                for _ in range(self.baseline_iterations):

                    # recreate original tree
                    temp = self.create_tree()

                    # fit copy of forest model for one EM step
                    temp.fit(features_, labels_)

                    # grab updated error for copy model
                    train_error_temp = self.eval_loss(
                        temp.predict(eval_features), eval_labels
                    )

                    # see if error has improved, update tree if so
                    if train_error_temp < train_error:

                        tree = copy.deepcopy(temp)
                        train_error = train_error_temp

                    if test_mode:
                        error_trajectory.append(
                            self.eval_loss(tree.predict(eval_features), eval_labels)
                        )

                # self.trees[tree_num] = tree
                train_error = self.eval_loss(tree.predict(eval_features), eval_labels)
                temp = copy.deepcopy(tree)

                # do EM iterations
                for _ in range(self.em_iterations - 1):

                    # fit copy of forest model for one EM step
                    temp.pb_train(features_, labels_)
                    temp.pb_refit(
                        features_, labels_, cur_node=temp.root, refit_all=True
                    )

                    # grab updated error for copy model
                    train_error_temp = self.eval_loss(
                        temp.predict(eval_features), eval_labels
                    )

                    # see if error has improved, update tree if so
                    if train_error_temp < train_error:

                        tree = copy.deepcopy(temp)
                        train_error = train_error_temp

                        if test_mode:
                            error_trajectory.append(
                                self.eval_loss(tree.predict(eval_features), eval_labels)
                            )

                    else:
                        if test_mode:
                            error_trajectory.append(
                                self.eval_loss(tree.predict(eval_features), eval_labels)
                            )

                    # make sure temp is up to date
                    temp = copy.deepcopy(tree)

                # last iteration there is no need to refit all split points, and no need to recopy self
                temp.pb_train(features_, labels_)
                temp.pb_refit(features_, labels_, cur_node=temp.root, refit_all=False)

                # grab updated error for copy model
                train_error_temp = self.eval_loss(
                    temp.predict(eval_features), eval_labels
                )

                # see if error has improved, update self if so
                if train_error_temp < train_error:

                    tree = copy.deepcopy(temp)

                # update metrics
                self.feature_importances += temp.feature_importances
                self.ave_split_points_changed += (
                    tree.split_points_changed / self.n_estimators
                )
                self.ave_split_features_changed += (
                    tree.split_features_changed / self.n_estimators
                )
                if tree.early_stop:
                    self.early_stops += 1

                # finally, append tree to self.trees
                self.trees[tree_num] = tree

                if test_mode:
                    error_trajectory.append(
                        self.eval_loss(tree.predict(eval_features), eval_labels)
                    )

                    if self.error_trajectory is None:
                        self.error_trajectory = np.zeros(len(error_trajectory))

                    self.error_trajectory += (
                        np.array(error_trajectory) / self.n_estimators
                    )

        # otherwise, run fit, pb_fit, and pb_refit at the forest level
        else:
            features_ = features
            labels_ = labels

            if not eval_labels:
                eval_labels = labels_

            if not eval_features:
                eval_features = features_

            # fit initial model
            self.fit(features_, labels_)

            temp = copy.deepcopy(self)
            train_error = self.eval_loss(self.predict(eval_features), eval_labels)

            # do EM iterations
            for i in range(self.em_iterations - 1):

                # fit copy of forest model for one EM step
                temp.pb_fit(features_, labels_)
                temp.pb_refit(features_, labels_, refit_all=True)

                # grab updated error for copy model
                train_error_temp = temp.eval_loss(
                    temp.predict(eval_features), eval_labels
                )

                # see if error has improved, update self if so
                if train_error_temp < train_error:

                    self.trees = copy.deepcopy(temp.trees)
                    self.feature_importances = copy.deepcopy(temp.feature_importances)
                    self.early_stops = copy.deepcopy(temp.early_stops)
                    self.ave_split_points_changed = copy.deepcopy(
                        self.ave_split_points_changed
                    )
                    self.ave_split_features_changed = copy.deepcopy(
                        self.ave_split_features_changed
                    )

                    train_error = train_error_temp

                # reset temp to be a copy of original again
                temp = copy.deepcopy(self)

            # last iteration there is no need to refit all split points, and no need to recopy self
            temp.pb_fit(features_, labels_)
            temp.pb_refit(features_, labels_, refit_all=False)

            # grab updated error for copy model
            train_error_temp = self.eval_loss(temp.predict(eval_features), eval_labels)

            # see if error has improved, update self if so
            if train_error_temp < train_error:

                self.trees = copy.deepcopy(temp.trees)
                self.feature_importances = copy.deepcopy(temp.feature_importances)
                self.early_stops = copy.deepcopy(temp.early_stops)
                self.ave_split_points_changed = copy.deepcopy(
                    self.ave_split_points_changed
                )
                self.ave_split_features_changed = copy.deepcopy(
                    self.ave_split_features_changed
                )

    def fit(self, features, labels, bootstrap=True):
        """
        generous forest method for initial fit of forest

        inputs:
                features: nxd array
                labels: nx1 array

        calls:
                basically everything not related to pb or clustering
        """

        self.total_features = features.shape[1]
        # only need to do these once to begin
        if self.label_map_pred is None:

            self.preprocess_labels(labels)
            self.organize_hyperparams()
            self.feature_importances = np.zeros(self.total_features)

        for i in range(self.n_estimators):

            # create tree
            tree = self.create_tree()

            # possible junk here...do we really need to maintain probailistic stuff for composition models?
            if bootstrap:

                features_, labels_, _ = GenerousForest.impose_bootstrap(
                    bootstrap, features, labels
                )

            else:
                features_ = features
                labels_ = labels

            tree.fit(features_, labels_)

            self.ave_tree_train_error += tree.train_error / self.n_estimators
            self.ave_leaves_per_tree += tree.leaves / self.n_estimators
            self.ave_nodes_per_tree += tree.nodes / self.n_estimators

            self.trees[i] = tree
            self.feature_importances += tree.feature_importances

    def predict(self, features, pb_features=False):
        """
        generous forest predict method

        inputs:
                features: nxd array
                pb_features: bool - whether features are compatitble with pb feature probs

        returns:
                nx1 array of predicted labels for input features
        """
        preds = np.empty((len(features), len(self.trees)), dtype=object)

        if pb_features:
            features_ = GenerousTree.pb_preprocess_features(features)

        else:
            features_ = features

        if len(self.trees) == 0:

            return preds

        for i in range(len(self.trees)):

            tree = self.trees[i]

            y_hat = tree.predict(features_, pb_features=pb_features)

            preds[:, i] = y_hat

        return self.vote_method(preds)

    def pb_fit(self, features, labels, warm_start=False, pb_features=False):
        """
        refit all trees using pb

        inputs:
                features:nxd array
                labels: nx1 array
                warm_start: bool - whether we are retraining a preexisting model
                pb_features: bool - whether features are compatitble with pb feature probs
        """

        if not pb_features:

            features_ = GenerousTree.pb_preprocess_features(features)

        else:
            features_ = features

        # reset feature importances
        self.feature_importances = np.zeros(self.total_features)

        for i in self.trees:

            i.pb_train(
                features_,
                labels,
                warm_start=warm_start,
                pb_features=True,
            )

            self.ave_split_features_changed += i.split_features_changed / len(
                self.trees
            )

            if i.early_stop:

                self.early_stops += 1

            self.feature_importances += i.feature_importances

    def pb_refit(self, features, labels, pb_features=False, refit_all=False):
        """
        generous forest method to refit pb tree with the best split point for each feature

        if using in EM strategy, refit_all should be set to True
        """

        if not pb_features:

            features = GenerousTree.pb_preprocess_features(features)

        for i in self.trees:

            i.pb_refit(
                features, labels, cur_node=i.root, pb_features=True, refit_all=refit_all
            )
            self.ave_split_points_changed += i.split_points_changed / len(self.trees)
