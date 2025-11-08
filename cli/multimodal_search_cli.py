import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image_path", type=str, help="The path of the image")

    image_search_parser = subparsers.add_parser("image_search", help="Search from image")
    image_search_parser.add_argument("image_path", type=str, help="The path of the image")

    args = parser.parse_args()

    match args.command:
        case "image_search":
            results = image_search_command(args.image_path)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['similarity_score']:.3f})") 
                print(f"   {result['description'][:100]}")
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
           parser.print_help() 


if __name__ == "__main__":
    main()
