![](gui/src/logo.svg)

FREUD (**F**eature **R**etrieval, **E**diting and **U**nderstanding for **D**evelopers) is a codebase for discovering and analyzing intermediate activations in audio models. It provides
- Code for training sparse autoencoders on audio model activations
- An interactive GUI for inspecting base model activations as well as learned autoencoder features

Currently, it is compatible with OpenAI's [Whisper](https://github.com/openai/whisper) family of models.

Checkpoints with the corresponding training run logs are available on [Huggingface](https://huggingface.co/collections/cherrvak/whisper-sparse-autoencoders-673bbfc58f51fde3c5b23754).
# Demo
You can demo the GUI [here](https://feature-demo.ksadov.com/). Input an MLP neuron index and see melspecs of audio that strongly activate that feature with that neuron's activation values overlaid, i.e strong activations for index 0 correspond to an "m" phoneme. You can also a record or upload short audio clip and see which features it activates most strongly.

# Setup
1. Create a virtual env. I used conda and python 3.10: `conda create -n whisper-interp python=3.10`
2. Activate your virtual env and install pytorch: `conda init whisper-interp; conda install pytorch -c pytorch`
3. Install the rest of the dependencies: `pip install -r requirements.txt`
4. Download the LibriSpeech datasets: `python -m src.scripts.download_audio_datasets`
5. Running the GUI requires [installing NodeJS](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) if it isn't already installed on your machine. Once installed, `cd` into the `gui` directory and run `npm install` to install GUI dependencies.
6. Install ffmpeg if it isn't already installed on your machine.

# General Notes
1. Config files in `configs/features` and `configs/train` specify `"cuda"` as the device, but you can change this field to `"cpu"` if you're not using an NVIDIA GPU.
2. If you're low on disk space:
  - You can set the `collect_max` parameter for configs in `config/features` to an integer amount to save activations for only that many files.
  - Collection steps are optional, at the cost of slower feature search and training:
    - For feature search, omit the `--from_disk` flag
    - For training, set the `from_disk` field in the training config to `false`.
3. Once you've started the GUI webserver, `cd` into the `gui` directory and run the command `npm run start`. The GUI will be displayed at `http://localhost:3000/`. The client code assumes that the GUI server is running on port 5555 of `localhost`, which will not be the case if you are running the server on a remote machine. In that case, edit the file `gui/src/ActivationDisplay.js` to set `API_BASE_URL` to the correct remote URL.
4. If you find activation search too slow, set the `--files_to_search` flag of `src.scripts.gui_server` to N in order to search through only N files in the dataset.
5. I look for interesting features by inputing clips to the Upload Audio tab of the GUI, making note of the top feature indexes for the uploaded clip and then checking if the pattern held for files returned by the Activation Search results for those indexes. I've found this to be a more productive (and fun!) than browsing indexes at random.

# Single-neuron interpretability
According to previous results, ["neurons in the MLP layers of the encoder are highly interpretable."](https://er537.github.io/blog/2023/09/05/whisper_interpretability.html). Follow the steps in this section to replicate the results of section 1.1 of the linked post.

1. Collect MLP activations from the speech dataset: `python -m src.scripts.collect_activations --config configs/features/tiny_block_2_mlp_1_test.json`
2. Start the GUI server and follow step 3 of #General Notes to view activations: `python -m src.scripts.gui_server --config configs/features/tiny_block_2_mlp_1.json --from_disk`

Interesting things to note:
- The top activations for the first 50 MLP neurons follow the pattern laid out in the linked section's table. However, when if you look at strongly *negative* activations by setting activation value to 0 and checking "use absolute value", you'll see that the most strongly negative activations are also appear to follow the same pattern!
- When MLP neurons correspond to features, those features tend to be phonetic rather than anything of broader semantic meaning. A few potential exceptions:
  - 1110 activates before pauses between words where you would a comma to appear in the transcript
  - 38 appears to activate most strongly at the start of an exclamation?

If you like, you can repeat the same steps above on the residual stream output of block 2 rather than just MLP activations. As per section 1.2 of the link, looking at single indices for these activations will fail to yield human-comprehensible features, though some correspond have maybe-interesting activation patterns:
- 85 alternates high and low on the scale of 0.7 seconds
- 232 has a strong negative activation roughly equidistant between strong positive activations at the silence before speech

# Training an L1-regularized sparse autoencoder on Whisper Tiny activations
These steps will train an sparse autoencoder dictionary for block 2 of Whisper Tiny, following the autoencoder architecture of [Interpreting OpenAI's Whisper](https://github.com/er537/whisper_interpretability/blob/master/whisper_interpretability/sparse_coding/train/train.py).

1. Collect block 2 activations for the train, validation and test datasets: `python -m src.scripts.collect_activations --config configs/features/tiny_block_2_train; python -m src.scripts.collect_activations --config configs/features/tiny_block_2_dev; python -m src.scripts.collect_activations --config configs/features/tiny_block_2_test;`
2. Train a SAE: `python -m src.scripts.train_sae --config configs/train/tiny_l1.json`
- Tensorboard training logs and checkpoints will be saved to the directory `runs/`
3. Once the run has completed to your satisfaction, you can collect trained SAE activations: `python -m src.scripts.collect_activations --config configs/features/tiny_l1_sae.json`
4. Start the GUI server and follow step 3 of General notes to view activations:: `python -m src.scripts.gui_server --config configs/features/tiny_l1_sae.json --from_disk`

# Training a k-sparse autoencoder on Whisper Tiny activations
These steps will train a sparse autoencoder based on [Eleuther AI's implementation of k-sparse autoencoders](https://github.com/EleutherAI/sae). It uses K-sparsity and AuxK loss introduced by [Gao et al. 2024](https://arxiv.org/abs/2406.04093v1) in order to combat dead dictionary entries.

1. Follow step 1 of the section above to (optionally) collect block 2 activations.
2. Train a SAE: `python -m src.scripts.train_sae --config configs/train/tiny_topk.json`
- See step 2 note above for logging and checkpoint information
3. After the run's completion, collect trained SAE activations: `python -m src.scripts.collect_activations --config configs/features/tiny_topk_sae.json`
4. Start the GUI server and follow step 3 of General notes to view activations:: `python -m src.scripts.gui_server --config configs/features/tiny_topk_sae.json --from_disk`

# Training an L1-regularized autoencoder on Whisper Large V3 activations
1. Collect block 16 activations for the train and validation datasets: `python -m src.scripts.collect_activations --config configs/features/large_v3_block_16_train_10k.json; python -m src.scripts.collect_activations --config configs/features/large_v3_block_16_dev`
- To economize disk space `configs/features/large_v3_block_16_train_10k.json` only collects activations for 10000 files, but you can alter that number in the config as you wish (or omit caching activations to disk altogether, see General Note 2)
3. Once the run has completed, collect trained SAE activations: `python -m src.scripts.collect_activations --config configs/features/large_v3_l1_sae.json`
4. Start the GUI server and follow step 3 of General notes to view activations: `python -m src.scripts.gui_server --config configs/features/large_v3_l1_sae.json --from_disk`

# Training an L1-regularized autoencoder on Whisper Large sound effect activations
[Gong et al. 2023](https://www.isca-archive.org/interspeech_2023/gong23d_interspeech.pdf) demonstrated that unlike most ASR models, Whisper Large encodes information about background noise deep into its intermediate representation. Following the paper, we train on the [AudioSet](https://research.google.com/audioset/) dataset and test on [ESC-50](https://github.com/karolpiczak/ESC-50). I found that L1-regularized SAE training to be unstable, so I trained a k-sparse one.

1. Download the AudioSet and ESC-50 datasets: `python -m src.scripts.download_audio_datasets --dataset audioset; python -m src.scripts.download_audio_datasets --dataset esc-50`
2. Collect activations: `python -m src.scripts.collect_activations --config configs/features/large_v1_block_16_audioset_train.json; python -m src.scripts.collect_activations --config configs/features/large_v1_block_16_audioset_train.json;`
3. Train a SAE: `python -m src.scripts.train_sae --config configs/train/tiny_topk.json`
4. Collect SAE activations: `python -m src.scripts.collect_activations --config configs/features/topk_large_v1_whisper-at.json`
5. Start the GUI server and follow step 3 of General notes to view activations: `python -m src.scripts.gui_server --config configs/features/topk_large_v1_whisper-at.json --from_disk`
