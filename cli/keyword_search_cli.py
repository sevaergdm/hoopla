import argparse

from lib.keyword_search import build_command, search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movie_list = search_command(args.query, 20)
            for i, movie in enumerate(movie_list, 1):
                print(f"{i}. {movie.get("title", "")}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Successfully built inverted index")
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
