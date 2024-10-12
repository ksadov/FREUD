import argparse
import os
import requests
import tarfile
from tqdm import tqdm


roots = {
    "librispeech": "https://www.openslr.org/resources/12",
    "audioset": "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data"
}
files = {
    "librispeech": [
        "test-other.tar.gz",
        "dev-other.tar.gz",
        "train-other-500.tar.gz",
    ],
    "audioset": [
        "balanced_train_segments.csv",
        "bal_train00.tar",
        "bal_train01.tar",
        "bal_train02.tar",
        "bal_train03.tar",
        "bal_train04.tar",
        "bal_train05.tar",
        "bal_train06.tar",
        "bal_train07.tar",
        "bal_train08.tar",
        "bal_train09.tar",
        "eval_segments.csv",
        "eval00.tar",
        "eval01.tar",
        "eval02.tar",
        "eval03.tar",
        "eval04.tar",
        "eval05.tar",
        "eval06.tar",
        "eval07.tar",
        "eval08.tar",
        "ontology.json",
    ]
}


def download_files(output_dir: str, dataset: str):
    """
    Download files corresponding to one of the available datasets (librispeech, audioset)

    :param output_dir: The directory to save the downloaded files (saved within a subdirectory named after the dataset)
    :param dataset: The dataset to download
    """
    os.makedirs(output_dir, exist_ok=True)
    filtered_files = [file for file in files[dataset]
                      if not os.path.exists(os.path.join(output_dir, file))]
    for file in tqdm(filtered_files):
        url = os.path.join(roots[dataset], file)
        output_file = os.path.join(output_dir, file)
        if not os.path.exists(output_file):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(output_file, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=1024), desc=file, unit="KB"):
                    if chunk:
                        f.write(chunk)
    print("All files downloaded to", output_dir)


def extract_files(file_dir: str):
    """
    Extract all tar files in the given directory and remove the tar files after extraction

    :param file_dir: The directory containing the tar files to extract
    """
    for file in tqdm(os.listdir(file_dir)):
        if ".tar" in file:
            file_path = os.path.join(file_dir, file)
            with tarfile.open(file_path) as tar:
                tar.extractall(file_dir)
            os.remove(file_path)
    print("All files extracted in", file_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="audio_data",
                        help="The directory to save the downloaded files")
    parser.add_argument("--dataset", type=str, default="librispeech",
                        help="The dataset to download (must be one of librispeech, audioset)")
    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir, args.dataset)
    if args.dataset not in roots:
        raise ValueError(f"Dataset {args.dataset} not found in {roots.keys()}")
    download_files(out_dir, args.dataset)
    extract_files(out_dir)


if __name__ == "__main__":
    main()
