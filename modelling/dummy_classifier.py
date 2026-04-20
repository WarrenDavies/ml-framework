from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, cross_val_predict

from modelling.registry import register


@register
def dummy_classifier(run_ts, x_values, y_values, strategy="most_frequent", seed=42):

    model = DummyClassifier(strategy=strategy, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_results = cross_validate(
        model,
        x_values,
        y_values,
        cv=cv,
        scoring=scoring
    )

    positive_class_results = {}
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        mean = round(scores.mean(), 3)
        std = round(scores.std(), 3)
        print(f"{metric}: {mean} ± {std}")
        positive_class_results[metric] = {
            "mean": mean,
            "std": std,
        }

    y_pred = cross_val_predict(model, x_values, y_values, cv=5)
    print(classification_report(y_values, y_pred))
    report = classification_report(y_values, y_pred, output_dict=True)

    return positive_class_results, report