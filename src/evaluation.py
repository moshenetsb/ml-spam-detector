import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve

LABELS = ["ham", "spam"]


def _get_fig_ax(ax=None, figsize=(5, 4.5)):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    return fig, ax


def _get_scores(model, X):

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None



def compute_metrics(y_true, y_pred, scores=None) -> dict:

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_spam": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_spam": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_spam": f1_score(y_true,
         y_pred, pos_label=1, zero_division=0),
    }

    if scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
        metrics["average_precision"] = average_precision_score(y_true, scores)

    return metrics



def plot_confusion_matrix(y_true, y_pred, labels=LABELS, title="Confusion matrix", ax=None, save_path=None):
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = _get_fig_ax(ax, figsize=(4.5, 4))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 13})
    fig.colorbar(disp.im_, ax=ax, shrink=0.8, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Real label")
    
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        
    return ax


def plot_roc_curve(model, X, y, name=None, ax=None, save_path=None):

    fig, ax = _get_fig_ax(ax, figsize=(5, 4.5))
    display = RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classificator")
    ax.set_title("ROC curve")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return float(display.roc_auc)


def plot_precision_recall_curve(model, X, y, name=None, ax=None, save_path=None):

    fig, ax = _get_fig_ax(ax, figsize=(5, 4.5))
    PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax, name=name)
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return ax


def plot_top_features(model, vectorizer, top_n=15, save_path=None):

    feature_names = np.array(vectorizer.get_feature_names_out())
    log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]  # spam - ham

    top_spam_idx = np.argsort(log_prob_diff)[-top_n:]
    top_ham_idx = np.argsort(log_prob_diff)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].barh(feature_names[top_spam_idx], log_prob_diff[top_spam_idx], color="#A32C29")
    axes[0].set_title(f"Top {top_n} words indicating SPAM")
    axes[0].set_xlabel("log P(word | spam) - log P(word | ham)")

    axes[1].barh(feature_names[top_ham_idx], -log_prob_diff[top_ham_idx], color="#2B6B4F")
    axes[1].set_title(f"Top {top_n} words indicating HAM")
    axes[1].set_xlabel("log P(word | ham) - log P(word | spam)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        
    return axes



def plot_learning_curve(estimator, X, y, cv=5, scoring="f1", train_sizes=None, title="Learning curve", ax=None, save_path=None):

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 6)

    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=train_sizes,
        shuffle=True, random_state=42,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = _get_fig_ax(ax, figsize=(6.5, 4.5))

    ax.plot(sizes, train_mean, "o-", color="#A32C29", label="Result on training set")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, color="#A32C29", alpha=0.15)

    ax.plot(sizes, val_mean, "o-", color="#2B6B4F", label="Cross-validation result")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, color="#2B6B4F", alpha=0.15)

    ax.set_xlabel("Number of training examples")
    ax.set_ylabel(f"Result ({scoring})")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return ax


def plot_train_vs_test_metrics(model, X_train, y_train, X_test, y_test, title="Training vs test", ax=None, save_path=None):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]

    train_scores = [
        accuracy_score(y_train, y_train_pred),
        precision_score(y_train, y_train_pred, zero_division=0),
        recall_score(y_train, y_train_pred, zero_division=0),
        f1_score(y_train, y_train_pred, zero_division=0),
    ]

    test_scores = [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred, zero_division=0),
        recall_score(y_test, y_test_pred, zero_division=0),
        f1_score(y_test, y_test_pred, zero_division=0),
    ]

    fig, ax = _get_fig_ax(ax, figsize=(6.5, 4.5))
    x = np.arange(len(metric_names))
    width = 0.35

    ax.bar(x - width / 2, train_scores, width, label="Training dataset", color="#6C8EBF")
    ax.bar(x + width / 2, test_scores, width, label="Test dataset", color="#A32C29")

    for i, (tr, te) in enumerate(zip(train_scores, test_scores)):
        gap = tr - te
        if gap > 0.03:
            ax.annotate(f"Δ={gap:.2f}", (x[i], max(tr, te) + 0.02), ha="center", fontsize=8, color="#A32C29")

    ax.set_xticks(x, labels=metric_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)

    return ax


def plot_model_comparison(
    results,
    metrics=("f1",),
    baseline_value=None,
    baseline_label=None,
    family_fn=None,
    family_colors=None,
    title="Models comparison",
    ax=None,
    save_path=None,
):

    results_sorted = sorted(results, key=lambda r: r[metrics[0]], reverse=True)
    names = [r["name"] for r in results_sorted]
    y_pos = np.arange(len(names))

    fig, ax = _get_fig_ax(ax, figsize=(1.5*8, max(4, 0.6 * len(names) + 1.5)))

    legend_handles, legend_labels = [], []

    if len(metrics) > 1:
        palette = ["#6C8EBF", "#A32C29", "#2B6B4F", "#97907E"]
        height = 0.8 / len(metrics)
        for i, metric in enumerate(metrics):
            values = [r[metric] for r in results_sorted]
            offset = (i - (len(metrics) - 1) / 2) * height
            ax.barh(y_pos + offset, values, height=height,
                    label=metric.capitalize(), color=palette[i % len(palette)])
            
    else:
        metric = metrics[0]
        values = [r[metric] for r in results_sorted]

        if family_fn and family_colors:
            colors = [family_colors.get(family_fn(r["name"]), "#97907E") for r in results_sorted]
            ax.barh(y_pos, values, color=colors)
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in family_colors.values()]
            legend_labels = list(family_colors.keys())

        else:
            ax.barh(y_pos, values, color="#A32C29")

    if baseline_value is not None:
        ax.axvline(
            baseline_value, color="black", linestyle="--", linewidth=1,
            label=baseline_label or f"Baseline = {baseline_value:.3f}",
        )

    ax.set_yticks(y_pos, labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Wynik")
    ax.set_title(title)

    if legend_handles:
        line_handles, line_labels = ax.get_legend_handles_labels()
        ax.legend(legend_handles + line_handles, legend_labels + line_labels, loc="lower right", fontsize=8)

    else:
        ax.legend(loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return ax



def evaluate_model(base_path: Path = Path(__file__).resolve().parent.parent / "data") -> dict:

    vectorized_dir = base_path / "vectorized"
    models_dir = base_path / "models"
    output_dir = base_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "spam_classifier_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Firstly run training.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorized_dir / "tfidf_vectorizer.pkl")
    X_train, X_test, y_train, y_test = joblib.load(vectorized_dir / "tfidf_splits.pkl")

    y_pred = model.predict(X_test)
    scores = _get_scores(model, X_test)
    metrics = compute_metrics(y_test, y_pred, scores)

    y_train_pred = model.predict(X_train)
    metrics["train_f1_spam"] = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    metrics["train_test_f1_gap"] = metrics["train_f1_spam"] - metrics["f1_spam"]

    print("  Metics on test dataset with training comparison:")
    for name, value in metrics.items():
        print(f"    {name:<20}: {value:.4f}")

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

    model_name = type(model).__name__

    plot_confusion_matrix(y_test, y_pred, save_path=output_dir / "confusion_matrix.png")
    plot_roc_curve(model, X_test, y_test, name=model_name, save_path=output_dir / "roc_curve.png")
    plot_precision_recall_curve(model, X_test, y_test, name=model_name, save_path=output_dir / "precision_recall_curve.png")
    plot_learning_curve(model, X_train, y_train, save_path=output_dir / "learning_curve.png")
    plot_train_vs_test_metrics(model, X_train, y_train, X_test, y_test, save_path=output_dir / "train_vs_test.png")

    if hasattr(model, "feature_log_prob_"):
        plot_top_features(model, vectorizer, save_path=output_dir / "top_features.png")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  Plots and metrics saved in: {output_dir}")

    return metrics


if __name__ == "__main__":
    evaluate_model()
