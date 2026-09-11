import os
from pypdf import PdfReader
from pptx import Presentation

DATA_FOLDER = "./data"
OUTPUT_FILE = "entire_data_content.txt"

def extract_pdf_text(file_path: str) -> str:
    """Extracts text from a PDF file page by page."""
    reader = PdfReader(file_path)
    text_blocks = []
    
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            text_blocks.append(f"--- Page {idx} ---")
            text_blocks.append(text.strip())
            
    return "\n".join(text_blocks)

def extract_ppt_text(file_path: str) -> str:
    """Extracts text and tables from PowerPoint presentation (.pptx)."""
    prs = Presentation(file_path)
    text_blocks = []

    for slide_idx, slide in enumerate(prs.slides, start=1):
        text_blocks.append(f"--- Slide {slide_idx} ---")
        
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    line = paragraph.text.strip()
                    if line:
                        text_blocks.append(line)
            elif shape.has_table:
                table_lines = []
                for row in shape.table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_lines.append(" | ".join(row_data))
                if table_lines:
                    text_blocks.append("[Table]:")
                    text_blocks.extend(table_lines)
                    
    return "\n".join(text_blocks)

def process_data_folder():
    """Scans the data folder and extracts text from supported files."""
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Directory '{DATA_FOLDER}' does not exist. Creating it now...")
        os.makedirs(DATA_FOLDER)
        print(f"Please place your PDF and PPTX files inside the '{DATA_FOLDER}' directory.")
        return

    files = [f for f in os.listdir(DATA_FOLDER) if not f.startswith(".")]
    if not files:
        print(f"No files found inside '{DATA_FOLDER}' directory.")
        return

    all_extracted_text = []

    for filename in files:
        file_path = os.path.join(DATA_FOLDER, filename)
        file_ext = filename.split(".")[-1].lower()

        print(f"Processing: {filename}...")

        try:
            if file_ext == "pdf":
                content = extract_pdf_text(file_path)
            elif file_ext in ["ppt", "pptx"]:
                content = extract_ppt_text(file_path)
            else:
                print(f"Skipping unsupported file format: {filename}")
                continue

            all_extracted_text.append(f"========================================")
            all_extracted_text.append(f"FILE: {filename}")
            all_extracted_text.append(f"========================================")
            all_extracted_text.append(content)
            all_extracted_text.append("\n\n")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Write merged text output to text file
    if all_extracted_text:
        final_output = "\n".join(all_extracted_text)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(final_output)
            
        print(f"\nDone! Extracted text saved to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No readable content was extracted.")

if __name__ == "__main__":
    process_data_folder()