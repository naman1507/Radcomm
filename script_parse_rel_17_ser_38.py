import os
import zipfile
from tika import parser

# Path to the folder containing the Word files
document_folder = r"D:\RADCOM - Copy\Chatbot_LLM\3GPP_docs\Rel_17\series_38"
output_folder = r"D:\RADCOM - Copy\Chatbot_LLM\3GPP_docs\Rel_17\series_38_parsed"
# Get a list of all Word files in the folder
word_files = [f for f in os.listdir(document_folder) if (f.endswith(".docx") or f.endswith(".doc"))]

# Function to parse text from a Word file using Apache Tika
def parse_word_file(word_file):
    word_path = os.path.join(document_folder, word_file)

    # Parse the Word document using Apache Tika
    parsed = parser.from_file(word_path)
    parsed_text = parsed['content']

    return parsed_text

# Parse each Word file and save the parsed text to a new folder
for word_file in word_files:
    print(f"Parsing contents of {word_file}...")
    parsed_text = parse_word_file(word_file)

    # Save the parsed text to a new file in the output folder
    output_file = os.path.join(output_folder, f"{os.path.splitext(word_file)[0]}_parsed.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(parsed_text)

print("Parsing completed.")
