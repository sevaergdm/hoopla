import argparse

from lib.semantic_search import embed_text, verify_embeddings, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser("embed_text", help="Create embeddings for the provided text")
    embed_text_parser.add_argument("text", type=str, help="The text used to create embeddings")

    subparsers.add_parser("verify_embeddings", help="Verify the embeddings from the movies.json file")

    args = parser.parse_args()

    match args.command:
        case "verify_embeddings":
            verify_embeddings()
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
