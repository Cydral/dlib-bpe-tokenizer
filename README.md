# Byte-Pair Encoding (BPE) Tokenizer for Dlib

This repository contains an implementation of the **Byte-Pair Encoding (BPE)** algorithm, designed to provide a tokenizer for the Dlib library. The BPE tokenizer is particularly useful for training Transformer-based models in natural language processing (NLP) tasks. It supports subword tokenization, handling of special tokens, and efficient encoding/decoding of text data.

## Features

- **Subword Tokenization**: Implements the BPE algorithm to build a vocabulary of subword units.
- **Special Tokens**: Supports special tokens like `<|endoftext|>`, `<|unk|>`, and `<|pad|>`.
- **Custom Vocabulary Size**: Allows training with a user-defined vocabulary size.
- **File and Directory Support**: Can load training data from files or directories.
- **Save/Load Model**: Save the trained tokenizer model and vocabulary to disk, and load them for reuse.
- **Encoding/Decoding**: Encode text into subword tokens and decode tokens back into text.

## Installation

To use this program, ensure you have the following dependencies installed:

- **Boost Libraries**: Required for program options and filesystem operations.
- **C++ Compiler**: A C++14-compatible compiler is recommended.

## Example Model: `dlib_t3k_base`

For reference, a pre-trained model named **`dlib_t3k_base`** is provided. This model formalizes a vocabulary of **3000 tokens** (excluding special tokens) and was trained on a **generalist English document corpus of 1 billion bytes**. You can use this model as a starting point for your NLP tasks or fine-tune it for specific applications.

To load the `dlib_t3k_base` model:

```cpp
bpe_tokenizer tokenizer;
tokenizer.load("dlib_t3k_base");
