import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_excel(r"C:\Users\Vikram\Desktop\Grievance system\complex_complaints_data.xlsx")
df.columns = ['complaint', 'department']
df.dropna(inplace=True)

# Encode labels
encoder = LabelEncoder()
df['department_encoded'] = encoder.fit_transform(df['department'])

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['complaint'])
y = df['department_encoded']

# Function to extract TP, FP, FN, TN for each class from the confusion matrix
def extract_confusion_matrix_metrics(cm, classes):
    metrics = {}
    for i, class_name in enumerate(classes):
        tp = cm[i, i]  # True Positive: diagonal element
        fn = cm[i, :].sum() - tp  # False Negative: sum of the column - TP
        fp = cm[:, i].sum() - tp  # False Positive: sum of the row - TP
        tn = cm.sum() - (tp + fn + fp)  # True Negative: remaining elements
        metrics[class_name] = {
            'True Positive (TP)': tp,
            'False Positive (FP)': fp,
            'False Negative (FN)': fn,
            'True Negative (TN)': tn
        }
    return metrics

# Function to save individual category images for TP, FP, FN, TN
def save_individual_category_images(metrics, model_name, split_name, categories):
    for category, metrics_values in metrics.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.bar(metrics_values.keys(), metrics_values.values(), color='skyblue')
        ax.set_title(f'{model_name} - {category} Confusion Matrix Metrics ({split_name} split)')
        ax.set_ylabel('Values')
        plt.tight_layout()
        plt.savefig(f'{model_name}{category}{split_name}_metrics.png')
        plt.close()

        # Save confusion matrix for this category
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        cm_category = cm[encoder.classes_ == category, encoder.classes_ == category]
        sns.heatmap(cm_category, annot=True, fmt='g', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name} Confusion Matrix ({category} - {split_name} split)')
        plt.tight_layout()
        plt.savefig(f'{model_name}{category}_confusion_matrix{split_name}.png')
        plt.close()

# Train-test split (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

# Train and evaluate Random Forest (optional comparison)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Calculate confusion matrices
log_cm = confusion_matrix(y_test, y_pred_log)
rf_cm = confusion_matrix(y_test, y_pred_rf)

# Extract metrics for Logistic Regression and Random Forest
log_metrics = extract_confusion_matrix_metrics(log_cm, encoder.classes_)
rf_metrics = extract_confusion_matrix_metrics(rf_cm, encoder.classes_)

# Display the metrics
print("\nLogistic Regression Confusion Matrix Metrics (TP, FP, FN, TN):")
print(log_metrics)
print("\nRandom Forest Confusion Matrix Metrics (TP, FP, FN, TN):")
print(rf_metrics)

# Save the better model (example: logistic regression)
best_model = log_model  # or rf_model if that performs better
pickle.dump(best_model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))

# Print classification report
print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_log))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# Plot confusion matrix for both models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Logistic Regression Confusion Matrix-like heatmap
sns.heatmap(pd.DataFrame(log_metrics).T[['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)']], annot=True, cmap='Blues', fmt='g', ax=axes[0], cbar=False)
axes[0].set_title("Logistic Regression Confusion Matrix Metrics")
axes[0].set_xlabel("Metrics")
axes[0].set_ylabel("Categories")

# Random Forest Confusion Matrix-like heatmap
sns.heatmap(pd.DataFrame(rf_metrics).T[['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)']], annot=True, cmap='Greens', fmt='g', ax=axes[1], cbar=False)
axes[1].set_title("Random Forest Confusion Matrix Metrics")
axes[1].set_xlabel("Metrics")
axes[1].set_ylabel("Categories")

plt.tight_layout()
plt.show()

# Plot accuracy comparison for both models
plt.bar(["Logistic Regression", "Random Forest"], [acc_log, acc_rf], color=['blue', 'green'])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (70-30 Split)")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

print(f" Logistic Regression Accuracy: {acc_log:.2f}")
print(f" Random Forest Accuracy: {acc_rf:.2f}")
print(" Best model saved successfully as model.pkl")

# Save confusion matrix tables to Excel
log_cm_table = pd.DataFrame(log_metrics).T
rf_cm_table = pd.DataFrame(rf_metrics).T

with pd.ExcelWriter('complex_complaints_data.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Complaints Data', index=False)
    log_cm_table.to_excel(writer, sheet_name='Logistic Regression Metrics', index=True)
    rf_cm_table.to_excel(writer, sheet_name='Random Forest Metrics', index=True)

print(" Confusion matrix and metrics saved to 'complex_complaints_with_metrics.xlsx'")