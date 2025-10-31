import argparse

from lib.evaluation import evaluate_result


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    results = evaluate_result(limit)

    print(f"k={limit}")
    print()
    for query, result in results["results"].items():
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {result["precision"]:.4f}")
        print(f"  - Recall@{limit}: {result["recall"]:.4f}")
        print(f"  - F1 Score: {result["f1"]:.4f}")
        print(f"  - Retrieved: {result["retrieved"]}")
        print(f"  - Relevant: {result["relevant"]}")
        print()


if __name__ == "__main__":
    main()
