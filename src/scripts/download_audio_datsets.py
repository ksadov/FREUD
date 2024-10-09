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
  ],
    "audioset": [
      "balanced_train_segments.csv",
      "bal_train00.tar",
      "bal_train01.tar",
      "eval_segments.csv",
      "eval00.tar",
      "ontology.json",
  ]
}

def download_files(output_dir: str, dataset: str):
    os.makedirs(output_dir, exist_ok=True)
    filtered_files = [file for file in files[dataset] if not os.path.exists(os.path.join(output_dir, file))]
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
    Extract and delete tar files in the directory
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
    parser.add_argument("--output_dir", type=str, default="audio_data")
    parser.add_argument("--dataset", type=str, default="librispeech")
    args = parser.parse_args()
    out_dir = os.path.join(args.output_dir, args.dataset)
    if args.dataset not in roots:
        raise ValueError(f"Dataset {args.dataset} not found in {roots.keys()}")
    download_files(out_dir, args.dataset)
    extract_files(out_dir)

if __name__ == "__main__":
    main()