import argparse

from lib.search_utils import (DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE,
                              DEFAULT_MAX_CHUNK_SIZE)
from lib.semantic_search import (chunk_text, embed_chunks, embed_query_text, embed_text,
                                 search_command, semantic_chunk_text, verify_embeddings,
                                 verify_model)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Create embeddings for the provided text"
    )
    embed_text_parser.add_argument(
        "text", type=str, help="The text used to create embeddings"
    )

    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings from the movies.json file"
    )

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Create embeddings from user query"
    )
    embedquery_parser.add_argument("query", type=str, help="The query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search for a given query with semantic search"
    )
    search_parser.add_argument("query", type=str, help="The query to search for")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="How many results to return"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk the input text")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="The chunk size"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="The number of overlapping words",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the input text semantically"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=DEFAULT_MAX_CHUNK_SIZE,
        help="The max chunk size",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="The number of overlapping sentences",
    )

    subparsers.add_parser("embed_chunks", help="Create chunked embedding")

    args = parser.parse_args()

    match args.command:
        case "embed_chunks":
            embed_chunks()
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "search":
            search_command(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
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
