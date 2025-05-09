import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

data = pd.read_csv("heart.csv")
X = data.drop("target", axis=1)
y = data["target"]

print("Class Distribution:\n", y.value_counts())

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
rf.fit(X_train, y_train)


rf_test_preds = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_test_preds)
rf_conf_matrix = confusion_matrix(y_test, rf_test_preds)
rf_report = classification_report(y_test, rf_test_preds)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("Random Forest Confusion Matrix:\n", rf_conf_matrix)
print("Random Forest Classification Report:\n", rf_report)

rf_train_probs = rf.predict_proba(X_train)[:, 1].reshape(-1, 1)
rf_test_probs = rf.predict_proba(X_test)[:, 1].reshape(-1, 1)

X_train_combined = np.hstack((X_train, rf_train_probs))
X_test_combined = np.hstack((X_test, rf_test_probs))

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train_combined, y_train)

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(gb, "model/gradient_boosting.pkl")

hybrid_model = {
    "scaler": scaler,
    "random_forest": rf,
    "gradient_boosting": gb
}
joblib.dump(hybrid_model, "model/hybrid_model.pkl")


y_pred = gb.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"\n Hybrid Model Accuracy: {accuracy:.4f}")
print("Hybrid Model Confusion Matrix:\n", conf_matrix)
print("Hybrid Model Classification Report:\n", report)
