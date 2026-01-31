import json
from openai_extractor import extract_with_openai
import evaluator
import time

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1"
]


def main():
    # Load dataset
    with open("dataset.json", "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)

    # Load ground truth
    with open("ground_truth.json", "r", encoding="utf-8-sig") as f:
        ground_truth = json.load(f)

    ground_truth_map = {item["chat_id"]: item for item in ground_truth}

    final_results = {}

    for model in OPENAI_MODELS:
        print(f"\n==============================")
        print(f"Running extraction with model: {model}")
        print(f"==============================")

        all_results = []
        latencies_ms = []   # ⬅️ NEW

        for item in dataset:
            chat_id = item["chat_id"]
            conversation = item["conversation"]

            print(f"\n\nProcessing {chat_id}")

            try:
                start_time = time.perf_counter()   # ⬅️ START

                prediction = extract_with_openai(
                    conversation=conversation,
                    model=model
                )

                end_time = time.perf_counter()     # ⬅️ END
                latency_ms = (end_time - start_time) * 1000
                latencies_ms.append(latency_ms)

                gt = ground_truth_map.get(chat_id)
                if not gt:
                    print(f"⚠️ Ground truth missing for {chat_id}")
                    continue

                norm_gt = evaluator.normalize_ground_truth(gt)
                norm_pred = evaluator.normalize_prediction(prediction)

                print('\nNormalized Ground truth data :' , json.dumps(norm_gt, indent=4))
                print('\n\nNormalized Prediction data :', json.dumps(norm_pred, indent=4))

                comparison = evaluator.compare(norm_pred, norm_gt)
                all_results.append(comparison)

            except Exception as e:
                print(f"❌ Error processing {chat_id}: {e}")

        # Compute metrics
        metrics = evaluator.compute_metrics(all_results)

        # ⬅️ ADD LATENCY METRICS
        if latencies_ms:
            metrics["latency"] = {
                "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
                "min_latency_ms": round(min(latencies_ms), 2),
                "max_latency_ms": round(max(latencies_ms), 2),
                "num_samples": len(latencies_ms)
            }
        else:
            metrics["latency"] = None

        final_results[model] = metrics

    print("\n=== FINAL METRICS (PER MODEL) ===")
    print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
        main()