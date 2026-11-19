"""Transient-stability classification of the IEEE 39-bus system with a random forest.

Modules
  config    file locations, feature names, split sizes, forest grid
  data      load, describe and split the dataset (train / validation / test)
  model     random-forest builder, tuning by grouped cross-validation, save / load
  evaluate  metrics, breakdowns, feature importance, learning curve
  plots     all charts
  predict   predictions for new cases
"""
