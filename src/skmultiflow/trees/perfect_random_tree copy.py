from operator import attrgetter
import numpy as np
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.gini_split_criterion import GiniSplitCriterion
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.utils.utils import get_dimensions, calculate_object_size

Node = HoeffdingTree.Node
SplitNode = HoeffdingTree.SplitNode
ActiveLearningNode = HoeffdingTree.ActiveLearningNode
InactiveLearningNode = HoeffdingTree.InactiveLearningNode

GINI_SPLIT = 'gini'
INFO_GAIN_SPLIT = 'info_gain'
MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'
error_width_threshold = 300


class PerfectRandomTree(HoeffdingTree):
    """ Hoeffding Anytime Tree or Extremely Fast Decision Tree.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=1)
        Number of instances a leaf should observe between split attempts.
    n_split_trial: int (default=10)
        Number of split trials in each node, each sample.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    Hoeffding Anytime Tree or Extremely Fast Decision Tree (EFDT) [1]_ constructs a tree incrementally. HATT seeks
    to select and deploy a split as soon as it is confident the split is useful, and then revisits that decision,
    replacing the split if it subsequently becomes evident that a better split is available. HATT learns rapidly from a
    stationary distribution and eventually it learns the asymptotic batch tree if the distribution from which the data
    are drawn is stationary.

    References
    ----------
    .. [1]  None

    """

    class PERTSplitNode(SplitNode):
        """ Node that splits the data in a PERT Tree.

        Parameters
        ----------
        split_test: InstanceConditionalTest
            Split test.
        class_observations: dict (class_value, weight) or None
            Class observations
        attribute_observers : dict (attribute id, AttributeClassObserver)
            Attribute Observers
        """

        def __init__(self, split_test, class_observations, attribute_observers):
            """ AnyTimeSplitNode class constructor."""
            super().__init__(split_test, class_observations)
            self._attribute_observers = attribute_observers
            self._weight_seen_at_last_split_reevaluation = 0
            return


    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=1,
                 n_split_trial=1,
                 nominal_attributes=None
                 ):

        super(PerfectRandomTree,self).__init__(max_byte_size=max_byte_size,
                                               memory_estimate_period=memory_estimate_period,
        )  

    # Override partial_fit
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Incrementally trains the model. Train samples (instances) are composed of X attributes and their
        corresponding targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: list or numpy.array
            Contains the class values in the stream. If defined, will be used to define the length of the arrays
            returned by `predict_proba`
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        Notes
        -----
        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the path from root to the corresponding leaf for the instance and
          sort the instance.

          * Reevaluate the best split for each internal node.
          * Attempt to split the leaf.
        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            for i in range(row_cnt):
                self._partial_fit(X[i], y[i])

    #  Override _partial_fit
    def _partial_fit(self, X, y):
        """ Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        y: int
            Class label for sample X.
        sample_weight: float
            Sample weight.

        """

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1

        # Sort instance X into a leaf
        self._sort_instance_into_leaf(X, y)




    def _sort_instance_into_leaf(self, X, y):
        """ Sort an instance into a leaf.

        Private function where leaf learn from instance:

        1. Find the node where instance should be.
        2. If no node have been found, create new learning node.
        3.1 Update the node with the provided instance

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """
        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
        leaf_node = found_node.node

        if leaf_node is None:
            leaf_node = self._new_learning_node()
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._active_leaf_node_cnt += 1

        if isinstance(leaf_node, self.LearningNode):
            learning_node = leaf_node
            learning_node.learn_from_instance(X, y, 1, self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self.estimate_model_byte_size()


    #  Override _attempt_to_split
    def _attempt_to_split(self, node, parent, branch_index):
        """ Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the best one.
        2. Compute the Hoeffding bound.
        3. If the difference between the best split candidate and the don't split candidate is larger than
        the Hoeffding bound:
            3.1 Replace the leaf node by a split node.
            3.2 Add a new leaf node on each branch of the new split node.
            3.3 Update tree's metrics

        Parameters
        ----------
        node: AnyTimeActiveLearningNode
            The node to reevaluate.
        parent: AnyTimeSplitNode
            The node's parent.
        branch_index: int
            Parent node's branch index.

        """

        if not node.observed_class_distribution_is_pure():

            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)

            if len(best_split_suggestions) > 0:

                # x_best is the attribute with the highest G_int
                best_split_suggestions.sort(key=attrgetter('merit'))
                x_best = best_split_suggestions[-1]

                # Get x_null
                x_null = node.get_null_split(split_criterion)

                # Force x_null merit to get 0 instead of -infinity
                if x_null.merit == -np.inf:
                    x_null.merit = 0.0

                hoeffding_bound = self.compute_hoeffding_bound(
                    split_criterion.get_range_of_merit(node.get_observed_class_distribution()), self.split_confidence,
                    node.get_weight_seen())

                if x_best.merit - x_null.merit > hoeffding_bound or hoeffding_bound < self.tie_threshold:

                    # Split
                    new_split = self.new_split_node(x_best.split_test,
                                                    node.get_observed_class_distribution(),
                                                    node.get_attribute_observers())

                    # update weights in
                    new_split.update_weight_seen_at_last_split_reevaluation()

                    for i in range(x_best.num_splits()):
                        new_child = self._new_learning_node(x_best.resulting_class_distribution_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += x_best.num_splits()

                    if parent is None:
                        # root case : replace the root node by a new split node
                        self._tree_root = new_split
                    else:
                        parent.set_child(branch_index, new_split)

                    # Manage memory
                    self.enforce_tracker_limit()


        # Override new_split_node
    def new_split_node(self, split_test, class_observations, attribute_observers):
        """ Create a new split node."""
        return self.AnyTimeSplitNode(split_test, class_observations, attribute_observers)