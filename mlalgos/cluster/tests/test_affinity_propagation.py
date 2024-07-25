import numpy as np

from mlalgos.cluster import AffinityPropagation


def test_affinity_propagation() -> None:
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

    clustering = AffinityPropagation(random_state=5)

    clustering.fit(X)

    expected_labels = np.array([0, 0, 0, 1, 1, 1])

    assert np.array_equal(clustering.labels, expected_labels)

    expected_cluster_centers = np.array([[1, 2], [4, 2]])
    assert np.array_equal(clustering.cluster_centers, expected_cluster_centers)

    prediction = clustering.predict([[0, 0], [4, 4]])
    expected_prediction = np.array([0, 1])

    assert np.array_equal(prediction, expected_prediction)
