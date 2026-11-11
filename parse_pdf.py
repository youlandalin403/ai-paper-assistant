import pymupdf

doc = pymupdf.open("paper.pdf")

print("PDF pages:", doc.page_count)

page_texts = []

for page in doc:
    text = page.get_text()
    page_texts.append(text)

page_text = page_texts[0]

print("page_text type:", type(page_text))
print("page 1 length:", len(page_text))

full_text = "\n".join(page_texts)

print("full_text type:", type(full_text))
print("full_text length:", len(full_text))

print("===== FIRST 1000 CHARACTERS =====")
print(full_text[:1000])

page = doc[1]

print("===== PAGE 2: get_text() =====")
print(page.get_text()[:1500])

print("===== PAGE 2: get_text(sort=True) =====")
print(page.get_text(sort=True)[:1500])

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

doc.close()