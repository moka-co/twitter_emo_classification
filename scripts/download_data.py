# This is a script

# Download glove
from tqdm import tqdm
import requests
import os
import zipfile

filepath="data/"

# Download the glove model as zip, if not already present, then unzip i
def download_glove():
    glove_url="http://nlp.stanford.edu/data/glove.6B.zip"
    glove_filepath = os.path.join(filepath, "glove.6B.zip")
    glove_model = os.path.join(filepath, "glove.6B.300d.txt")

    if not os.path.exists(glove_filepath):
        try:
            response = requests.get(glove_url, stream=True, timeout=10)
            response.raise_for_status()  # raise for bad status

            total_size = int(response.headers.get('content-length', 0))
            print(f"Total_size= total_size")

            filename= os.path.join(filepath, "glove.6B.zip")
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit="iG",
                unit_scale=True,
                unit_divisor=128
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):
                    size = f.write(chunk)
                    progress_bar.update(size)

            print(f"Downloaded {filename} succesfully")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")

    if not os.path.exists(glove_model):
        with zipfile.ZipFile(glove_filepath, 'r') as zip_ref:
            # Extract all files to a specified directory (or current directory if none provided)
            zip_ref.extractall(filepath)
            print(f"Extraction complete! Files are in: {filepath}")

    print(f"Successfully download and extracted Glove model under {filepath} directory")


# Run functions
download_glove()