import numpy as np
import pandas as pd

from preprocessing import split_features_and_target
from classifiers import NaiveBayes

def test_naive_bayes() -> None:
    data = [
        ("Rainy", "Hot", "High", "f", "no"),
        ("Rainy", "Hot", "High", "t", "no"),
        ("Overcast", "Hot", "High", "f", "yes"),
        ("Sunny", "Mild", "High", "f", "yes"),
        ("Sunny", "Cool", "Normal", "f", "yes"),
        ("Sunny", "Cool", "Normal", "t", "no"),
        ("Overcast", "Cool", "Normal", "t", "yes"),
        ("Rainy", "Mild", "High", "f", "no"),
        ("Rainy", "Cool", "Normal", "f", "yes"),
        ("Sunny", "Mild", "Normal", "f", "yes"),
        ("Rainy", "Mild", "Normal", "t", "yes"),
        ("Overcast", "Mild", "High", "t", "yes"),
        ("Overcast", "Hot", "Normal", "f", "yes"),
        ("Sunny", "Mild", "High", "t", "no")
    ]

    df = pd.DataFrame(data, columns=["Outlook", "Temp", "Humidity", "Windy", "Play"])

    X, y = split_features_and_target(df, "Play")

    clf = NaiveBayes()
    clf.fit(X, y)

    expected_likelihoods = {
        'Outlook': {
            'Overcast': {
                'no': 0.0, 
                'yes': 0.4444444444444444
            },
            'Rainy': {
                'no': 0.6, 
                'yes': 0.2222222222222222
            },
            'Sunny': {
                'no': 0.4, 
                'yes': 0.3333333333333333
            }
        },
        'Temp': {
            'Cool': {
                'no': 0.2, 
                'yes': 0.3333333333333333
            },
            'Hot': {
                'no': 0.4, 
                'yes': 0.2222222222222222
            },
            'Mild': {
                'no': 0.4, 
                'yes': 0.4444444444444444
            }
        },
        'Humidity': {
            'High': {
                'no': 0.8, 
                'yes': 0.3333333333333333
            },
            'Normal': {
                'no': 0.2, 
                'yes': 0.6666666666666666
            }
        },
        'Windy': {
            'f': {
                'no': 0.4, 
                'yes': 0.6666666666666666
            },
            't': {
                'no': 0.6, 
                'yes': 0.3333333333333333
            }
        }
    }

    assert clf._likelihoods == expected_likelihoods

    X_test = [
        ["Rainy", "Mild", "Normal", "t"],
        ["Overcast", "Cool", "Normal", "t"],
        ["Sunny", "Hot", "High", "t"]
    ]

    results = clf.predict(X_test)
    labels = ["yes", "yes", "no"]

    assert results == labels