=================================================================
Permutation Importance with Multicollinear or Correlated Features
=================================================================

In this example, we compute the
:func:`~sklearn.inspection.permutation_importance` of the features to a trained
:class:`~sklearn.ensemble.RandomForestClassifier` using the
:ref:`breast_cancer_dataset`. The model can easily get about 97% accuracy on a
test dataset. Because this dataset contains multicollinear features, the
permutation importance shows that none of the features are important, in
contradiction with the high test accuracy.

We demo a possible approach to handling multicollinearity, which consists of
hierarchical clustering on the features' Spearman rank-order correlations,
picking a threshold, and keeping a single feature from each cluster.

.. note::
    See also
    :ref:`sphx_glr_auto_examples_inspection_plot_permutation_importance.py

    presence of correlated features, as demonstrated in the following section.


# %%
# Random Forest Feature Importance on Breast Cancer Data
# ------------------------------------------------------
#
# First, we define a function to ease the plotting:

`labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
    # renamed to `tick_labels`. The following code handles this, but as a
    # scikit-learn user you probably can write simpler code by using `labels=...`
    # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).


# %%
# The plot on the left shows the Gini importance of the model. As the
# scikit-learn implementation of
# :class:`~sklearn.ensemble.RandomForestClassifier` uses a random subsets of
# :math:`\sqrt{n_\text{features}}` features at each split, it is able to dilute
# the dominance of any single correlated feature. As a result, the individual
# feature importance may be distributed more evenly among the correlated
# features. Since the features have large cardinality and the classifier is
# non-overfitted, we can relatively trust those values.
#
# The permutation importance on the right plot shows that permuting a feature
# drops the accuracy by at most `0.012`, which would suggest that none of the
# features are important. This is in contradiction with the high test accuracy
# computed as baseline: some feature must be important.
#
# Similarly, the change in accuracy score computed on the test set appears to be
# driven by chance:
#
# Handling Multicollinear Features
# --------------------------------
# When features are collinear, permuting one feature has little effect on the
# models performance because it can get the same information from a correlated
# feature. Note that this is not the case for all predictive models and depends
# on their underlying implementation.
#
# One way to handle multicollinear features is by performing hierarchical
# clustering on the Spearman rank-order correlations, picking a threshold, and
# keeping a single feature from each cluster. First, we plot a heatmap of the
# correlated features:
