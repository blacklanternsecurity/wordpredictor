# Word Predictor

## Word Prediction Tool

This project provides a simple tool for training and testing a lightweight prediction model based on analyzing word frequencies using [word2vec](https://www.tensorflow.org/text/tutorials/word2vec).

Utilized in [BBOT](https://github.com/blacklanternsecurity/bbot) within the [ffuf_shortnames](https://github.com/blacklanternsecurity/bbot/blob/dev/bbot/modules/ffuf_shortnames.py) module. 

---

## Features

- **Train a Model**: Train a Word2Vec model on a text corpus and extract word frequency data into a lightweight format.
- **Predict Words**: Retrieve likely word completions for a given prefix using a given pre-trained model, ranked by frequency.


---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. 

### Prerequisites
- Python 3.9 or higher
- Poetry (install with `pip install poetry` if not already installed)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/word-predictor.git
   cd word-predictor
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

   Or run with poetry run:
   ```
   poetry run python3 wordpredictor.py
   ```

---

## Usage

The tool supports two modes: `train` and `test`.

### 1. **Training Mode**
Train a Word2Vec-based word predictor on a custom text file.

#### Command
```bash
poetry run word-predictor train <file_path> [--min_count <value>] [--debug]
```

#### Arguments
- `<file_path>`: Path to the input text file containing words (one per line).
- `--min_count`: Minimum frequency for a word to be included in the vocabulary (default: 2).
- `--debug`: Enable debug mode to print tokens during training.

#### Example
```bash
poetry run word-predictor train words.txt --min_count 5
```

---

### 2. **Testing Mode**
Test the predictor by retrieving predictions for a given prefix.

#### Command
```bash
poetry run word-predictor test <model_path> --prefix <prefix> --n <top_n>
```

#### Arguments
- `<model_path>`: Path to the trained `.pred` file.
- `--prefix`: Prefix to predict words for.
- `--n`: Number of top predictions to retrieve.

#### Example
```bash
poetry run word-predictor test words.pred --prefix "pre" --n 5
```