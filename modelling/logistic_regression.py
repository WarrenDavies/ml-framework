import datetime
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from modelling.registry import register


@register
def logistic_regression(run_ts, x_values, y_values, x_cols, random_state=42, max_iter=1000, splits=5):

    model = LogisticRegression(random_state=random_state, max_iter=max_iter)
    model.fit(x_values, y_values)

    scaler = StandardScaler()
    x_values = scaler.fit_transform(x_values)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
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


    ## Feature Importance

    odds_change = [round(math.e ** coef, 3) for coef in model.coef_[0]]
    coefs = [round(coef, 3) for coef in model.coef_[0]]
    df_feature_importance = (
        pd.DataFrame({
            'Feature': x_cols,
            'Coefficient': coefs,
            'Odds_change': odds_change
        })
        .sort_values(by='Coefficient', key=lambda x: x.abs(),  ascending=False)
        # .sort_values(by='Coefficient', ascending=False)
        .reset_index()
        .drop("index", axis=1)
    )
    print("\nFeature Importance:\n", df_feature_importance)

    
    ### charts
    chart_paths = []

    # Sigmoid
    log_odds = model.decision_function(x_values)
    probs = model.predict_proba(x_values)[:, 1]
    sorted_idx = np.argsort(log_odds)
    log_odds_sorted = log_odds[sorted_idx]
    probs_sorted = probs[sorted_idx]
    y_jitter = y_values + np.random.normal(0, 0.02, size=len(y_values))
    plt.scatter(log_odds, y_jitter, alpha=0.3)
    plt.plot(log_odds_sorted, probs_sorted, linewidth=2, label="Sigmoid")
    plt.xlabel("Log-Odds (Model Output)")
    plt.ylabel("Class / Probability")
    plt.title("Logistic Regression Fit")
    plt.legend()
    chart_path = f"outputs/charts/lr_sigmoid_{run_ts}.png"
    chart_paths.append(chart_path) 
    plt.savefig(chart_path)
    plt.close()


    # Confusion Matrix
    fig, ax = plt.subplots()

    y_pred = model.predict(x_values)
    cm = confusion_matrix(y_values, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)

    ax.set_title("Confusion Matrix")

    chart_path = f"outputs/charts/confusion_matrix_{run_ts}.png"
    chart_paths.append(chart_path)

    plt.savefig(chart_path)
    plt.close(fig)

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay.from_predictions(
        y_values, y_pred, normalize="true", ax=ax
    )

    ax.set_title("Normalised Confusion Matrix")

    chart_path = f"outputs/charts/confusion_matrix_normalised_{run_ts}.png"
    chart_paths.append(chart_path)

    plt.savefig(chart_path)
    plt.close(fig)



    continuous_variables = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    for i, continuous_variable in enumerate(continuous_variables):
        pred_probs = model.predict_proba(x_values)[:, 1]
        pred_probs = np.clip(pred_probs, 0.001, 0.999)
        log_odds = np.log(pred_probs / (1 - pred_probs))

        feature_index = i 
        feature_name = continuous_variables[i]

        sns.regplot(x=x_values[:,i], y=log_odds, lowess=True, line_kws={'color': 'red'})
        plt.xlabel(feature_name)
        plt.ylabel('Log-Odds (Logit)')
        plt.title(f'Linearity Check: {feature_name} vs. Log-Odds')
        chart_path = f"outputs/charts/{feature_name}_{run_ts}.png"
        chart_paths.append(chart_path) 
        plt.savefig(chart_path)
        plt.close()

    return {
        "positive_class_results": positive_class_results,
        "classification_report": report,
        "chart_paths": chart_paths,
        "feature_importance": df_feature_importance,
    } 

    # print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # feature_importance = pd.DataFrame({
    #     'Feature': [f'Feature_{i}' for i in range(X.shape[1])],
    #     'Coefficient': model.coef_[0]
    # }).sort_values(by='Coefficient', ascending=False)
    # print("\nFeature Importance:\n", feature_importance)
