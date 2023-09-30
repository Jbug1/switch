class GenerousNode:
    """
    nodes that will make up the generous tree

    attributes:
                left: node: handling values less than split points
                right: node: handling values geq split point
                attribute: int: index of column from features that this node splits on (index of feature)
                split_point: numeric: x< split_point -> left, geq split_point -> right
                samples: array
    """

    def __init__(self, parent, depth):

        # mandatory attributes
        self.parent = parent
        self.depth = depth

        # fit attributes
        self.impurity = None
        self.num_instances = 0
        self.left = None
        self.right = None
        self.split_points = None
        self.split_feature = None
        self.split_features = None
        self.split_point = None
        self.probabilities = None
        self.feature_probs = None
        self.split_prob = None
        self.pred = None
        self.pred_prob = None
        self.freqs = None

        # pb attributes
        self.feature_probs_ = None
        self.split_points_ = None
        self.split_features_ = None
        self.active_feature_index = None
        self.times_traversed = 0
