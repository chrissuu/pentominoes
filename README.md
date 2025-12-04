# 15-816 Final Project: SAT and Pentomino Fences

## Installation

1. SBVA is provided as a submodule. After cloning the repository, run `git submodule update --init --recursive` to
   initialize and update the submodule. Then, follow the instructions in the `SBVA/README.md` file to build SVBA.
2. This project uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. Install all Python dependencies by
   first [installing uv](https://docs.astral.sh/uv/getting-started/installation/), then running `uv sync`.

After installation, you can run any Python script with `uv run <script_name>`.

## Usage

### Running tests

Run all unit tests with `uv run pytest`.
