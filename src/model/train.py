
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent))
import sys
from pathlib import Path

# Add the parent directory of the current file's parent to sys.path
sys.path.append(str(Path(__file__).parent.parent))


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
from ingestion.ingest_data import load_data
from preprocessing.preprocess import preprocess


df = load_data()
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("iris_classifier")
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    print(f"Test Accuracy: {acc}")