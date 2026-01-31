import re
import json


def normalize(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return value

EVAL_FIELDS = [
    "full_name",
    "email",
    "phone",
    "budget_rupees",
    "current_location",
    "preferred_location",
    "buying_timeline_weeks",
    "profession",
    "visit_date"
]

def normalize_prediction(pred):
    full_name = None
    if pred.get("first_name"):
        full_name = pred["first_name"]
        if pred.get("last_name"):
            full_name += " " + pred["last_name"]

    return {
        "full_name": full_name,
        "email": pred.get("email"),
        "phone": pred.get("phone_number"),
        "budget_rupees": pred.get("budget"),
        "current_location": pred.get("current_location"),
        "preferred_location": pred.get("preferred_location"),
        "buying_timeline_weeks": pred.get("buying_timeline_weeks"),
        "profession": pred.get("profession"),
        "visit_date": pred.get("visit_date"),
    }

def timeline_to_weeks(timeline):
    if not timeline:
        return None

    timeline = timeline.lower().strip()

    # extract all numbers
    nums = list(map(int, re.findall(r"\d+", timeline)))
    if not nums:
        return None

    avg = sum(nums) / len(nums)

    if "month" in timeline:
        return int(avg * 4)   # approx conversion
    if "week" in timeline:
        return int(avg)

    return None

def normalize_ground_truth(gt):
    entities = gt["entities"]

    return {
        "full_name": entities.get("customer_name"),
        "email": entities.get("email"),
        "phone": entities.get("phone"),
        "budget_rupees": (
            int(entities["budget_crore"] * 1e7)
            if entities.get("budget_crore") is not None
            else None
        ),
        "current_location": entities.get("current_location"),
        "preferred_location": entities.get("location"),
        "buying_timeline_weeks": timeline_to_weeks(
            entities.get("purchase_timeline")
        ),
        "profession": entities.get("profession"),
        "visit_date": entities.get("visit_date"),
    }
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

def compare(pred_norm, gt_norm):
    results = {}
    for key in pred_norm:
        results[key] = int(pred_norm[key] == gt_norm[key])
    print('\n#### Comparison (GT VS Pred) ####')
    print(json.dumps(results, indent=4))
    return results


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
