import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# এই নামটির দিকে খেয়াল করুন (create_visualizations)
def create_visualizations(y_test, y_pred, model, X_columns):
    # Confusion Matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('AI Cybersecurity: Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    # Feature Importance
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X_columns)
    importances.nlargest(10).plot(kind='barh', color='skyblue')
    plt.title('Top 10 Features for Threat Detection')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()