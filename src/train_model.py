"""
Trains a Random Forest model on the synthetic land cover dataset and saves the model and evaluation results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

def create_classification_map(clf, size=(100, 100)):
    simulated_grid = np.random.uniform(0, 1, (size[0], size[1], 5))
    flat = simulated_grid.reshape(-1, 5)
    preds = clf.predict(flat)
    mapping = {'forest': 0, 'oil_palm': 1, 'cocoa': 2, 'urban': 3, 'water': 4}
    reshaped = np.array([mapping[p] for p in preds]).reshape(size)
    plt.imshow(reshaped, interpolation='nearest')
    plt.title('Simulated Classified Land Cover Map')
    plt.axis('off')
    Path('outputs').mkdir(exist_ok=True, parents=True)
    plt.savefig('outputs/classified_map.png', bbox_inches='tight')
    plt.close()

def main():
    # Load data
    df = pd.read_csv('data/synthetic_land_cover.csv')
    X = df[['NDVI', 'NIR', 'RED', 'BLUE', 'SWIR']]
    y = df['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    Path('models').mkdir(exist_ok=True, parents=True)
    joblib.dump(clf, 'models/rf_model.pkl')

    # Save report
    with open('models/model_metrics.txt', 'w') as f:
        f.write(report)

    # Save confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(clf.classes_))
    plt.xticks(tick_marks, clf.classes_, rotation=45)
    plt.yticks(tick_marks, clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    Path('outputs').mkdir(exist_ok=True, parents=True)
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

    # Save feature importance
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh', title='Feature Importances')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()

    # Generate simulated map
    create_classification_map(clf)

    print("✅ Model training complete. Results saved in 'models/' and 'outputs/'.")

if __name__ == '__main__':
    main()
