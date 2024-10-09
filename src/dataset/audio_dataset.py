
import os
import torch

from src.utils.audio_utils import get_mels_from_audio_path, is_audio_file

class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for audio files. Returns a tuple of mel spectrogram and audio filename, where mel spectrogram is None if 
    calculate_mel is False.
    """
    def __init__(self, audio_folder: str, device: torch.device, calculate_mel: bool = True):
        super().__init__()
        self.audio_folder = audio_folder
        self.audio_files = self._get_audio_file_list()
        self.device = device
        self.calculate_mel = calculate_mel

    def _get_audio_file_list(self) -> list[str]:
        audio_files = []
        for root, dirs, files in os.walk(self.audio_folder):
            for file in files:
                if is_audio_file(file):
                    # check if root is global path
                    if not os.path.isabs(root):
                        root = os.path.abspath(root)
                    audio_files.append(os.path.join(root, file))
        return audio_files
    
    def __getitem__(self, idx) -> dict:
        audio_filename = self.audio_files[idx]
        if self.calculate_mel:
            mel = get_mels_from_audio_path(self.device, audio_filename)
        else:
            mel = None
        return mel, audio_filename
    
    def __len__(self) -> int:
        return len(self.audio_files)

def test_audio_dataset():
    device = torch.device("cpu")
    dataset = AudioDataset(
        "/Users/ksadov/Documents/ml_experiments/whisper_sae/librispeech_data/LibriSpeech/test-other", device)
    print(dataset[0])


if __name__ == "__main__":
    test_audio_dataset()
