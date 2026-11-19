# Structural probe CST 

An implementation of [Hewitt et al. 2019](https://aclanthology.org/N19-1419/) that applies the structural probe method to concrete syntax trees of code fragments across multiple programming languages. 

Our data is taken from [The Stack](https://huggingface.co/datasets/bigcode/the-stack), where we select roughly 1000 function definitions across the following programming languages: 

- Python 
- C
- C++ 
- Java
- JavaScript 
- PHP
- Go 
- Rust
- Haskell 

To parse the files, and extract function bodies, we use [tree-sitter](https://tree-sitter.github.io/tree-sitter/) due to its speed and extensive language support. 

## Progress 

- [x] Data processing 
    - [x] Download data 
    - [x] Parse each sample 
    - [x] Extract functions 
    - [x] Create distance matrix 
    - [x] Create depth vector 
    - [x] Hidden representations aligned with leaf nodes of CST 
- [ ] Training 
    - [x] Distance probe 
    - [ ] Depth probe 
    - [ ] Baseline 
- [ ] Evaluation 
    - [ ] Metrics 
- [ ] Analysis
    - [ ] Visualisations 
    - [ ] Ablations 

## Files 

- `src/download-dataset.py`: Responsible for downloading and creating code fragments, saves them to `trees.json` 
- `src/prepare-data.py`: Responsible for creating aligned hidden representations of the tokens in a given model using the data from `trees.json` 
- `data/`: Workspace package for creating and working with Concrete Syntax Trees
- `notebooks/`: Jupyter notebooks used for prototyping and testing 