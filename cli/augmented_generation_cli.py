import argparse

from lib.augmented_generation import citations_command, question_command, rag_command, summarize_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query to summarize results of"
    )
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results to summarize"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Get a list of citations for the query"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results to cite"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a user question"
    )
    question_parser.add_argument("query", type=str, help="User question")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results")

    args = parser.parse_args()

    match args.command:
        case "question":
            results = question_command(args.query)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("Answer:")
            print(results["answer"])
        case "citations":
            results = citations_command(args.query)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("LLM Answer:")
            print(results["citations"])
        case "summarize":
            results = summarize_command(args.query)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("LLM Summary:")
            print(results["summary"])
        case "rag":
            results = rag_command(args.query)
            print("Search Results:")
            for result in results["search_results"]:
                print(f"- {result['title']}")
            print("RAG Response:")
            print(results["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
