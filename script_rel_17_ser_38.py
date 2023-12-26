#script to dowmload doc files from 3GPP url

import os
import requests
from zipfile import ZipFile
from io import BytesIO
from bs4 import BeautifulSoup

# URL containing zip folders
url = "https://www.3gpp.org/ftp/Specs/2023-06/Rel-17/38_series"

# Function to extract Word documents from a zip file
def extract_word_docs(zip_url, output_folder):
    response = requests.get(zip_url)
    with ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall(output_folder)

# Function to iterate through zip folders and extract Word documents
def process_zip_folders(url, output_folder):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) >= 2 and 'href' in cells[1].a.attrs:
            zip_url = cells[1].a['href']
            print(f"Processing {zip_url}...")
            extract_word_docs(zip_url, output_folder)

# Output folder to store extracted Word documents
output_folder = "./3GPP_docs/Rel_17/series_38"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process zip folders and extract Word documents
process_zip_folders(url, output_folder)

print("Extraction complete.")

