import requests
import gdown
import os
import re
import subprocess
import time
from tqdm import tqdm

# ----- CONFIG -----
API_KEY = "YOUR_API_KEY"           # put your API key here
FOLDER_ID = "FOLDER_ID"       # Google Drive folder ID
OUTPUT_DIR = "assets/data/sd"           # local download folder
FILE_EXTENSION = ".zip"            # filter pth files
# -------------------

LIST_URL = "https://www.googleapis.com/drive/v3/files"

download = ['SD_pretrained.zip']

def list_files_in_folder(folder_id):
    global API_KEY
    all_files = []
    page_token = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id, name, mimeType)",
            "pageSize": 1000,
            "key": API_KEY
        }
        if page_token:
            params["pageToken"] = page_token

        r = requests.get("https://www.googleapis.com/drive/v3/files", params=params)
        r.raise_for_status()
        data = r.json()

        all_files.extend(data.get("files", []))

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return all_files


def download_drive_large_file(file_id, dest_path, filename):
    """
    Download a large Google Drive file that triggers the virus scan warning.
    """

    out_path = os.path.join(dest_path, filename)

    # 1. First request to get confirm token (can also parse HTML, here we know it’s 't')
    confirm = "t"

    # 2. Construct real download URL
    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}"

    # 3. Use curl to download actual file
    cmd = ["curl", "-L", "-o", out_path, download_url]
    subprocess.run(cmd, check=True)

    print(f"Downloaded {out_path} successfully.")


def fetch_files_and_download():
    global FOLDER_ID, FILE_EXTENSION, OUTPUT_DIR
    print("Fetching file list from Google Drive...\n")

    files = list_files_in_folder(FOLDER_ID)

    # filter to only .zip files
    zip_files = [f for f in files if f["name"].lower().endswith(FILE_EXTENSION)]

    print(f"Found {len(zip_files)} chekpoint files.\n")

    for f in zip_files:
        file_id = f["id"]
        filename = f["name"]

        if filename not in download:
            print(f"Skipping {filename}")
            continue

        download_drive_large_file(file_id, OUTPUT_DIR, filename)
        time.sleep(0.3)   # avoid hammering Google
        

    print("\nAll downloads completed.")


if __name__ == "__main__":
    fetch_files_and_download()
