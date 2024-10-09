1. Create a virtual env. I used conda and python 3.10: `conda create -n whisper-interp python=3.10`
2. Activate your virtual env and install pytorch: `conda init whisper-interp; conda install pytorch -c pytorch`
2. Install the rest of the dependencies: `pip install -r requirements.txt`
3. Download the speech dataset: `python -m src.scripts.download_audio_datasets`
4. Collect activations from the speech dataset: `python -m src.scripts.collect_activations --config whisper-interp/src/configs/features/block_2_mlp_1.json`
- if you're low on disk space, set `collect_max` or omit the collection step altogether
5. View the collected activations: `python -m src.scripts.gui_server --config whisper-interp/src/configs/features/block_2_mlp_1.json --from_disk`
- exclude the `--from_disk` flag to search activations generated on-the-fly