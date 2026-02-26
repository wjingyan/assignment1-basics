# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.
uv pip install scalene


### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
#### Updated command to download data
```
cd data
rm -f TinyStoriesV2-GPT4-train.txt TinyStoriesV2-GPT4-valid.txt owt_train.txt.gz owt_valid.txt.gz

# Re-download with the -L flag to follow redirects
# Download TinyStories
curl -L -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download and decompress OpenWebText sample
curl -L -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
gunzip -f owt_train.txt.gz
curl -L -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

cd ..
```
# Commands
# Part 2
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_train_bpe.py::test_train_bpe_speed
uv run cs336_basics/tokenizer.py
uv run cs336_basics/resource_monitor.py
uv run cs336_basics/run_train_bpe.py --dataset tinystories --split train
uv run cs336_basics/run_train_bpe.py --dataset owt --split valid
# Profile
# Profile the script running on the TinyStories validation set
uv run python -m cProfile -s cumulative cs336_basics/run_train_bpe.py --dataset tinystories --split validation
# Profile the script using Scalene
uv run python -m scalene run cs336_basics/run_train_bpe.py --dataset test --split train
# Profile a specific test using Scalene
uv run python -m scalene --pytest tests/test_train_bpe.py::test_train_bpe
# View Scalene results in browser
uv run scalene view
# tokenizer_experiments (a)
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_tinystories_train.json \
    --merges tokenizer_output/merges_tinystories_train.json \
    --input data/tinystory-sample.txt \
    --output output/encoding/encoded_tinystories.txt
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_owt_valid.json \
    --merges tokenizer_output/merges_owt_valid.json \
    --input data/owt-sample.txt \
    --output output/encoding/encoded_owt.txt
# tokenizer_experiments (b)
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_tinystories_train.json \
    --merges tokenizer_output/merges_tinystories_train.json \
    --input data/owt-sample.txt \
    --output output/encoding/encoded_owt_w_ts_tokenizer.txt
# tokenizer_experiments (d)
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_tinystories_train.json \
    --merges tokenizer_output/merges_tinystories_train.json \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output output/encoding/encoded_tinystories_train.npy
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_tinystories_train.json \
    --merges tokenizer_output/merges_tinystories_train.json \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output output/encoding/encoded_tinystories_valid.npy
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_owt_valid.json \
    --merges tokenizer_output/merges_owt_valid.json \
    --input data/owt_train.txt \
    --output output/encoding/encoded_owt_train.npy
uv run cs336_basics/run_tokenizer.py \
    --vocab tokenizer_output/vocab_owt_valid.json \
    --merges tokenizer_output/merges_owt_valid.json \
    --input data/owt_valid.txt \
    --output output/encoding/encoded_owt_valid.npy
# Part 3
uv run pytest -k test_linear
