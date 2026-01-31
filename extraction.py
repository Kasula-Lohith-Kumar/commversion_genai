import json
from openai_extractor import extract_with_openai
import evaluator

OPENAI_MODEL = "gpt-4.1-mini"  # change to gpt-4o / gpt-4.1 etc

def main():
    with open("dataset.json", "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)

    with open("ground_truth.json", "r", encoding="utf-8-sig") as f:
        ground_truth = json.load(f)

        # Convert ground truth list → dict keyed by chat_id
        ground_truth_map = {item["chat_id"]: item for item in ground_truth }
        print(ground_truth_map)

    all_results = []

    for item in dataset:
        chat_id = item["chat_id"]
        conversation = item["conversation"]

        print(f"Processing {chat_id}")

        prediction = extract_with_openai(conversation, model=OPENAI_MODEL)
        gt = ground_truth_map[chat_id]

        print("gt:", gt)

        norm_gt = evaluator.normalize_ground_truth(gt)
        norm_pred = evaluator.normalize_prediction(prediction)

        comparison = evaluator.compare(norm_pred, norm_gt)
        all_results.append(comparison)

    metrics = evaluator.compute_metrics(all_results)

    print("\n=== FINAL METRICS ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
