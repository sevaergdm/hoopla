import argparse

from lib.search_utils import DEFAULT_K_VALUE
from lib.hybrid_search import normalize_scores, rrf_search_command, weighted_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="The list of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="The query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight value")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="The number of results to return")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="The query")
    rrf_search_parser.add_argument("--k", type=int, default=DEFAULT_K_VALUE, help="The k parameter")
    rrf_search_parser.add_argument("--limit", type=int, help="The number of results to return")

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            results = rrf_search_command(args.query, args.k, args.limit)
            for i, result in enumerate(results["results"], 1):
                print(f"{i}. {result["title"]}")
                print(f"   RRF Score: {result.get("score", 0):.3f}")
                metadata = result.get("metadata", {})
                if "bm25_rank" in metadata and "semantic_rank" in metadata:
                    print(f"   BM25 Rank: {metadata["bm25_rank"]}, Semantic Rank: {metadata["semantic_rank"]}")
                print(f"   {result["document"][:100]}...")
                print()
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, result in enumerate(results["results"], 1):
                print(f"{i}. {result["title"]}")
                print(f"   Hybrid Score: {result.get("score", 0):.3f}")
                metadata = result.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(f"   BM25: {metadata["bm25_score"]:.3f}, Semantic: {metadata["semantic_score"]:.3f}")
                print(f"   {result["document"][:100]}...")
                print()
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for score in norm_scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
