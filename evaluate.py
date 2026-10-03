from sklearn.metrics import precision_recall_fscore_support

def evaluate_ner():
    # Ground truth vs predicted entities
    y_true = ["Disease", "Symptom", "Treatment", "Disease", "Symptom", "Clinical_event"]
    y_pred = ["Disease", "Symptom", "Treatment", "Symptom", "Symptom", "Clinical_event"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print("--- NER Model Evaluation Results ---")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-Score:  {f1:.2f}")

if __name__ == "__main__":
    evaluate_ner()