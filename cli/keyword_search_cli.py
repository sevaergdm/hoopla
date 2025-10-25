import argparse

from lib.keyword_search import build_command, idf_command, search_command, tf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for a given gocument ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="The term to search for")

    idf_parser = subparsers.add_parser("idf", help="Calculate the Inverse Document Frequency for a given term")
    idf_parser.add_argument("term", type=str, help="The term to search for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF score for a given doc id and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="The term to search for")


    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_list = search_command(args.query, 5)
            for movie in movie_list:
                print(f"{movie.get("id", "")}. {movie.get("title", "")}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Successfully built inverted index")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
