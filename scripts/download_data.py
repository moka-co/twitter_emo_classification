# This is a script

# Download glove
from tqdm import tqdm
import requests
import os
import zipfile
import shutil

# Create a header to mimic a real browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# Download the glove model as zip, if not already present, then unzip i
def download_glove():
    filepath="data/glove/"
    glove_url="http://nlp.stanford.edu/data/glove.6B.zip"
    glove_filepath = os.path.join(filepath, "glove.6B.zip")
    glove_model = os.path.join(filepath, "glove.6B.300d.txt")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

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


def download_semeval_dataset():
    filepath = "data/datasets"
    semeval_filepath_zip = os.path.join(filepath, "SemEval2018-Task1-all-data.zip")
    semeval_url = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"

    sem_eval_path_train = os.path.join(filepath, "SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-train.txt")
    sem_eval_path_val = os.path.join(filepath, "SemEval2018-Task1-all-data/English/E-c/2018-E-c-En-test-gold.txt")

    check_file = os.path.join(filepath, "2018-E-c-En-train.txt")
    if os.path.exists(check_file):
        print(f"Semeval dataset already present under {filepath}")
        return

    # Create filepath if it doesn't exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Create dataset file path if it does not exists
    if not os.path.exists(semeval_filepath_zip):
        try:
            response = requests.get(semeval_url, headers=headers, stream=True, timeout=10)
            response.raise_for_status()  # raise for bad status

            # Save zip file
            with open(semeval_filepath_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    size = f.write(chunk)

            # Extract zip file
            with zipfile.ZipFile(semeval_filepath_zip,'r') as zip_ref:
                zip_ref.extractall(filepath)

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")

        # Remove unecessary noise
        shutil.copy2(sem_eval_path_train, filepath)
        shutil.copy2(sem_eval_path_val, filepath)
        try:
            trash_dir = os.path.join(filepath, "__MACOSX")
            shutil.rmtree(trash_dir)
            trash_dir = os.path.join(filepath, "SemEval2018-Task1-all-data")
            shutil.rmtree(trash_dir)
            os.remove(semeval_filepath_zip)
            
        except Exception as e:
            print(f"Error {e} during cleanup")


    print(f"Successfully downloaded Semeval Dataset under {filepath}")


def download_eltea17():
    eltea_url = "https://raw.githubusercontent.com/RoozbehBandpey/ELTEA17/refs/heads/main/datasets/train.txt"
    filepath ="data/datasets"
    eltea_filepath = os.path.join(filepath, "eltea_train.txt")

    try:
        response = requests.get(eltea_url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()  # raise for bad status

        with open(eltea_filepath, 'wb') as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")

    print(f"Successfully downloaded ELTEA17 Dataset under {filepath}")



# Run functions
download_glove()
download_semeval_dataset()
download_eltea17()