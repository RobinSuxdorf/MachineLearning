import numpy as np

from mlalgos.cluster import KMeans


def test_kmeans() -> None:
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    kmeans = KMeans(n_clusters=2)

    kmeans.fit(X)

    expected_labels = np.array([1, 1, 1, 0, 0, 0])

    assert np.array_equal(expected_labels, kmeans.labels)

    expected_cluster_centers = np.array([[10, 2], [1, 2]])

    assert np.array_equal(expected_cluster_centers, kmeans.cluster_centers)

    prediction = kmeans.predict([[0, 0], [12, 3]])
    expected_prediction = np.array([1, 0])

    assert np.array_equal(expected_prediction, prediction)
