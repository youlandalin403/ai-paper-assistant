def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        target_end = start + chunk_size

        if target_end >= len(text):
            end = len(text)
        else:
            candidate = text[start:target_end]
            period_index = candidate.rfind(".")

            if period_index != -1:
                end = start + period_index + 1
            else:
                end = target_end

        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


if __name__ == "__main__":
    with open("output.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    print("number of chunks:", len(chunks))
    print("chunk 0 length:", len(chunks[0]))
    print("chunk 1 length:", len(chunks[1]))
    print("last chunk length:", len(chunks[-1]))

    print("\n===== CHUNK 0 END =====")
    print(chunks[0][-200:])

    print("\n===== CHUNK 1 START =====")
    print(chunks[1][:300])
