
import os
import torch
import torchaudio
import torch.nn.functional as F
from typing import Optional, Union
from subprocess import CalledProcessError, run
import numpy as np
import whisper

from functools import lru_cache

from constants import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_SAMPLES


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def trim_audio(
    array: np.array,
    start_time: float,
    end_time: float,
    sample_rate: int = 16_000,
):
    """
    Trim the audio file base array to n_samples, as expected by the encoder.
    """
    start_frame = int(sample_rate * start_time)
    end_frame = int(sample_rate * end_time)

    return array[start_frame:end_frame]


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def get_mels_from_audio_path(
    device, audio_path: str, start_time_s: Optional[float] = None, end_time_s: Optional[float] = None
):
    audio = load_audio(audio_path)
    if start_time_s is not None and end_time_s is not None:
        audio = trim_audio(audio, start_time_s, end_time_s)
    audio = pad_or_trim(audio.flatten())
    mels = log_mel_spectrogram(audio, device=device)
    return mels


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH,
                      window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, librispeech_folder: str, subset: str, device: torch.device, calculate_mel: bool = True):
        super().__init__()
        if not os.path.exists(librispeech_folder):
            download = True
        else:
            download = False
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            download=download, url=subset, root=librispeech_folder)
        self.root = librispeech_folder
        self.device = device
        self.calculate_mel = calculate_mel

    def __getitem__(self, idx) -> dict:
        file_name, sr, transcript, speaker_id, chapter_id, utterance_id = self.dataset.get_metadata(
            idx)
        global_file_name = os.path.join(self.root, "LibriSpeech", file_name)
        if self.calculate_mel:
            mel = get_mels_from_audio_path(self.device, global_file_name)
        else:
            mel = None
        return mel, utterance_id, global_file_name, transcript

    def __len__(self) -> int:
        return len(self.dataset)


def test_librispeech_dataset():
    device = torch.device("cuda")
    dataset = LibriSpeechDataset(
        "/home/ksadov/whisper_sae_dataset", "dev-other", device)
    print(dataset[0].mel.shape)


if __name__ == "__main__":
    test_librispeech_dataset()
