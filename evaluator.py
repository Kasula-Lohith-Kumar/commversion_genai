def normalize(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return value


def compare_prediction(pred, gt):
    """
    Returns dict like:
    {
      "first_name": True,
      "budget": False
    }
    """
    result = {}

    for key in gt.keys():
        result[key] = normalize(pred.get(key)) == normalize(gt.get(key))

    return result


def compute_metrics(all_results):
    tp = 0
    fn = 0

    for chat in all_results:
        for _, correct in chat.items():
            if correct:
                tp += 1
            else:
                fn += 1

    precision = tp / (tp + fn + 1e-9)
    recall = precision
    f1 = precision

    return {
        "true_positives": tp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }
