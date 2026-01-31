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
    nums = [int(x) for x in timeline.replace("weeks", "").split("-")]
    return sum(nums) // len(nums)

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
