import argparse

from lib.evaluation import llm_evaluation
from lib.hybrid_search import (normalize_scores, rrf_search_command,
                               weighted_search_command)
from lib.search_utils import DEFAULT_K_VALUE


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="The list of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="The query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight value"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="The number of results to return"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search"
    )
    rrf_search_parser.add_argument("query", type=str, help="The query")
    rrf_search_parser.add_argument(
        "--k", type=int, default=DEFAULT_K_VALUE, help="The k parameter"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="The number of results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method"
    )
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate search result"
    )

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            results = rrf_search_command(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )

            if results["enhanced_query"]:
                print(
                    f"Enhanced query ({results['enhanced_method']}): '{results['original_query']}' -> '{results['enhanced_query']}'\n"
                )

            if results["reranked"]:
                print(
                    f"Reranking top {len(results['results'])} results using {results['rerank_method']} method...\n"
                )

            if args.evaluate:
                eval = llm_evaluation(args.query, results["results"])
                for i, (res, score) in enumerate(zip(results["results"], eval), 1):
                    print(f"{i}. {res['title']}: {score}/3")

            for i, result in enumerate(results["results"], 1):
                print(f"{i}. {result["title"]}")
                if "individual_score" in result:
                    print(
                        f"   Rerank Score: {result.get('individual_score', 0):.3f}/10"
                    )
                if "batch_rank":
                    print(f"   Rerank Rank: {result.get('batch_rank', 0)}")
                if "cross_encoder_score" in result:
                    print(f"   Cross Encoder Score: {result.get('cross_encoder_score', 0):3f}")
                print(f"   RRF Score: {result.get("score", 0):.3f}")
                metadata = result.get("metadata", {})
                if "bm25_rank" in metadata and "semantic_rank" in metadata:
                    print(
                        f"   BM25 Rank: {metadata["bm25_rank"]}, Semantic Rank: {metadata["semantic_rank"]}"
                    )
                print(f"   {result["document"][:100]}...")
                print()
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, result in enumerate(results["results"], 1):
                print(f"{i}. {result["title"]}")
                print(f"   Hybrid Score: {result.get("score", 0):.3f}")
                metadata = result.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata["bm25_score"]:.3f}, Semantic: {metadata["semantic_score"]:.3f}"
                    )
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
