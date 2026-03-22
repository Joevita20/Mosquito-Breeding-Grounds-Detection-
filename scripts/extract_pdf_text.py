import pypdf
import sys

def extract_text(pdf_path, output_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    f.write(text + "\n")
        print(f"Successfully extracted text to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_pdf_text.py <pdf_path> <output_path>")
    else:
        extract_text(sys.argv[1], sys.argv[2])
