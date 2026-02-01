import re
import json
from datetime import datetime


# =================================================
# BASIC STRING NORMALIZATION
# =================================================

def normalize_str(value):
    if value is None:
        return None
    value = value.strip().lower()
    return value if value else None


# =================================================
# NAME NORMALIZATION
# =================================================

def split_name(full_name):
    """
    Capture first and last name ONLY if explicitly stated.
    """
    if not full_name:
        return None, None

    parts = full_name.strip().split()
    if len(parts) >= 2:
        return normalize_str(parts[0]), normalize_str(parts[-1])

    return normalize_str(parts[0]), None


# =================================================
# BUYING TIMELINE NORMALIZATION (LOWER BOUND)
# =================================================

def timeline_to_weeks(timeline):
    if not timeline:
        return None

    timeline = timeline.lower().strip()

    if "immediate" in timeline or "immediately" in timeline:
        return 0
    if "day after tomorrow" in timeline:
        return 0
    if "next month" in timeline:
        return 4
    if "next year" in timeline:
        return 52

    nums = list(map(int, re.findall(r"\d+", timeline)))
    if not nums:
        return None

    lower = min(nums)

    if "day" in timeline:
        return lower // 7
    if "week" in timeline:
        return lower
    if "month" in timeline:
        return lower * 4
    if "year" in timeline:
        return lower * 52

    return None


# =================================================
# BUDGET NORMALIZATION (GT — CORRECT)
# =================================================

def normalize_budget_from_gt(entities):
    """
    IMPORTANT:
    - Use ONLY explicit budget fields
    - DO NOT use price_mentioned_crore
    """

    if entities.get("budget_crore") is None:
        return None

    # budget_crore is already the customer's budget
    return int(float(entities["budget_crore"]) * 10_000_000)


# =================================================
# BUDGET NORMALIZATION (PREDICTION)
# =================================================

def normalize_budget_from_pred(budget):
    """
    Budget must be INR.
    Apply lower bound only if a range is given.
    """
    if budget is None:
        return None

    if isinstance(budget, (int, float)):
        return int(budget)

    nums = list(map(float, re.findall(r"\d+\.?\d*", str(budget))))
    if not nums:
        return None

    return int(min(nums))


# =================================================
# PROFESSION NORMALIZATION
# =================================================

def normalize_profession(profession):
    if not profession:
        return None

    profession = profession.lower()
    if profession in ["service", "business", "retired"]:
        return profession

    return None


# =================================================
# VISIT DATE NORMALIZATION
# =================================================

def normalize_visit_date(date_text):
    """
    Capture only explicitly mentioned dates in 2026.
    """
    if not date_text:
        return None

    try:
        dt = datetime.strptime(date_text, "%Y-%m-%d")
        if dt.year != 2026:
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


# =================================================
# GROUND TRUTH NORMALIZATION
# =================================================

def normalize_ground_truth(gt):
    entities = gt.get("entities", {})

    first_name, last_name = split_name(entities.get("customer_name"))

    return {
        "first_name": first_name,
        "last_name": last_name,
        "phone_number": normalize_str(entities.get("phone")),
        "email": normalize_str(entities.get("email")),
        "budget": normalize_budget_from_gt(entities),
        "current_location": normalize_str(entities.get("current_location")),
        "preferred_location": normalize_str(entities.get("location")),
        "profession": normalize_profession(entities.get("profession")),
        "visit_date": normalize_visit_date(entities.get("visit_date")),
        "buying_timeline_weeks": timeline_to_weeks(
            entities.get("purchase_timeline")
        ),
    }


# =================================================
# PREDICTION NORMALIZATION
# =================================================

def normalize_prediction(pred):
    return {
        "first_name": normalize_str(pred.get("first_name")),
        "last_name": normalize_str(pred.get("last_name")),
        "phone_number": normalize_str(pred.get("phone_number")),
        "email": normalize_str(pred.get("email")),
        "budget": normalize_budget_from_pred(pred.get("budget")),
        "current_location": normalize_str(pred.get("current_location")),
        "preferred_location": normalize_str(pred.get("preferred_location")),
        "profession": normalize_profession(pred.get("profession")),
        "visit_date": normalize_visit_date(pred.get("visit_date")),
        "buying_timeline_weeks": pred.get("buying_timeline_weeks"),
    }


# =================================================
# COMPARISON
# =================================================

def compare(pred_norm, gt_norm):
    results = {}
    for key in gt_norm:
        results[key] = int(pred_norm.get(key) == gt_norm.get(key))

    print("\n#### Comparison (GT vs Pred) ####")
    print(json.dumps(results, indent=4))
    return results


# =================================================
# METRICS
# =================================================

def compute_metrics(all_results):
    correct = 0
    total = 0

    for chat in all_results:
        for _, value in chat.items():
            total += 1
            correct += value

    return {
        "total_fields": total,
        "correct_fields": correct,
        "accuracy": round(correct / (total + 1e-9), 4)
    }
