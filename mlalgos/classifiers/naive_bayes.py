from typing import Any
import numpy as np
import pandas as pd


class NaiveBayes:
    """
    Naive Bayes classifier for categorical data.
    """

    def __init__(self) -> None:
        """
        Initializes the NaiveBayes classifier.
        """
        self._features: list[str] = []
        self._outcomes: np.ndarray = np.array([])
        self._prior_probabilities: dict[Any, float] = {}
        self._likelihoods: dict[str, dict[Any, dict[Any, float]]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fits the NaiveBayes classifier to the training data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target values.
        """
        self._features = list(X.columns)
        self._outcomes = np.unique(y)

        self._calc_prior_probabilities(X, y)
        self._calc_likelihoods(X, y)

    def _calc_prior_probabilities(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculates prior probabilities for each class.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target values.
        """
        class_counts = y.value_counts().to_dict()
        total_count = len(y)
        self._prior_probabilities = {
            outcome: count / total_count for outcome, count in class_counts.items()
        }

    def _calc_likelihoods(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Calculates likelihoods for each feature value given each class.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target values.
        """
        for feature in self._features:
            self._likelihoods[feature]: dict[Any, dict[Any, float]] = {}

            feature_counts = X.groupby([feature, y]).size().unstack(fill_value=0)

            for feature_value in feature_counts.index:
                self._likelihoods[feature][feature_value]: dict[Any, float] = {}

                for outcome in self._outcomes:
                    outcome_count = sum(y == outcome)
                    self._likelihoods[feature][feature_value][outcome] = (
                        feature_counts.at[feature_value, outcome] / outcome_count
                    )

    def predict(self, X: list[list[Any]]) -> list[Any]:
        """
        Predicts the class labels for the given data.

        Args:
            X (list[list[Any]]): Data to predict.

        Returns:
            list[Any]: Predicted class labels.
        """
        results: list[Any] = []

        for query in X:
            probs_outcome: dict[Any, float] = {}
            for outcome in self._outcomes:
                prior = self._prior_probabilities[outcome]
                likelihood = 1

                for feature, feature_value in zip(self._features, query):
                    likelihood *= self._likelihoods[feature][feature_value][outcome]

                probs_outcome[outcome] = prior * likelihood

            result = max(probs_outcome, key=probs_outcome.get)
            results.append(result)
        return results
