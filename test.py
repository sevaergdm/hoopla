from cli.lib.search_utils import load_movies, tokenize


def main():
    movies = load_movies()
    text = ""
    for movie in movies:
        id = movie.get("id", 0)
        if id != 3470:
            continue
        if id == 3470:
            text = f"{movie.get('title', '')}{movie.get('description')}"
            break

    tokens = tokenize(text)
    print(tokens)


if __name__ == "__main__":
    main()
