import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(importance: pd.DataFrame, path: str, top_n: int = 100):
    """
    Plot feature importance and write to image

    Args:
        importance:
            Dataframe which has "feature" and "importance" column
        path:
            File path to be saved
        top_n:
            Number of features visualized
    """
    sorted = importance.groupby('feature')['importance'].mean().reset_index().sort_values(by='importance', ascending=False)

    if len(sorted) > top_n:
        sorted = sorted.iloc[:top_n, :]

    plt.figure(figsize=(10, 16))
    sns.barplot(x="importance", y="feature", data=sorted)
    plt.title('Feature Importance (avg over folds)')
    plt.tight_layout()
    plt.savefig(path)
