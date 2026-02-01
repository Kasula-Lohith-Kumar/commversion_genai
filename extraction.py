import json
from openai_extractor import extract_with_openai
import evaluator
import time

OPENAI_MODELS = [
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-5-mini"
]


def main(prompt_file):
    """
    Evaluate multiple OpenAI models on a conversation extraction task by comparing 
    their predictions against ground truth data.

    This function:
    - Loads a dataset of conversations and corresponding ground truth labels
    - Runs structured extraction using different OpenAI models
    - Normalizes both predictions and ground truth
    - Computes evaluation metrics using a custom evaluator
    - Measures per-sample inference latency
    - Tracks token usage across all model calls
    - Aggregates and reports final performance metrics per model

    Args:
        prompt_file (str): 
            Path to the file containing the prompt template/instructions 
            to be used for extraction with the OpenAI models.

    Side effects:
        - Prints detailed progress information, normalized predictions/ground truth,
          and final metrics table to stdout
        - Accumulates token usage and latency statistics

    Output (printed):
        JSON-formatted final metrics dictionary containing, for each model:
        - Standard evaluation metrics (from evaluator.compute_metrics)
        - Latency statistics (avg/min/max latency in ms, number of samples)
        - Token usage statistics (total prompt, completion, total tokens,
          average tokens per sample)

    Returns:
        None
            Results are printed rather than returned.

    Raises:
        FileNotFoundError: If dataset.json, ground_truth.json or prompt_file cannot be opened
        json.JSONDecodeError: If JSON files are malformed
        Exception: Various errors from OpenAI API calls or normalization/comparison logic

    Example:
        >>> main("prompts/extraction_v2.txt")
        # Will print progress and final metrics table for all models
    """
    # Load dataset
    with open("dataset.json", "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)

    # Load ground truth
    with open("ground_truth.json", "r", encoding="utf-8-sig") as f:
        ground_truth = json.load(f)

    ground_truth_map = {item["chat_id"]: item for item in ground_truth}

    final_results = {}

    token_summary = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
        }

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

                result = extract_with_openai(
                conversation=conversation,
                model=model,
                pf=prompt_file
                )

                prediction = result["prediction"]
                usage = result["usage"]

                token_summary["prompt_tokens"] += usage["prompt_tokens"]
                token_summary["completion_tokens"] += usage["completion_tokens"]
                token_summary["total_tokens"] += usage["total_tokens"]

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

        metrics["tokens"] = {
            "total_prompt_tokens": token_summary["prompt_tokens"],
            "total_completion_tokens": token_summary["completion_tokens"],
            "total_tokens": token_summary["total_tokens"],
            "avg_tokens_per_sample": 
            round(token_summary["total_tokens"] / len(latencies_ms), 2) if latencies_ms else 0}

        final_results[model] = metrics

    print("\n=== FINAL METRICS (PER MODEL) ===")
    print(json.dumps(final_results, indent=2))



if __name__ == "__main__":
    ''' Main Function call execution with different 
            prompt files iteratively '''

    # prompt_files = ['prompt1.txt', 
    #                 'prompt2.txt', 
    #                 'prompt3.txt', 
    #                 'prompt4.txt',
    #                 'prompt5.txt',
    #                 'prompt6.txt']

    prompt_files = ['prompt1.txt', 'prompt2.txt', 'prompt3.txt']
    # prompt_files = ['prompt3.txt']

    for file in prompt_files:
        print(f"{20*'%'} Evaluating prompt {file} {20*'%'}")
        main(file)
        print(f"{20*'%'} End of Evauation of {file} {20*'%'}")