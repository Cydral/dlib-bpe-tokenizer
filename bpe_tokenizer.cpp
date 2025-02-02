/*
 * BPE Tokenizer Implementation for Dlib Library

 * This program implements the Byte-Pair Encoding (BPE) algorithm, a subword tokenization method
 * commonly used in natural language processing tasks, particularly for training Transformer models.
 * The BPE algorithm iteratively merges the most frequent pairs of bytes or characters to build a
 * vocabulary of subword units. This implementation supports training on text data, encoding text
 * into subword tokens, decoding tokens back into text, and saving/loading the tokenizer model.

 * Key Features:
 * - Supports training on text data from files or directories.
 * - Allows customization of the target vocabulary size.
 * - Handles special tokens such as <text>...</text>, <unk>, and <pad>.
 * - Enforces a maximum token length to prevent excessively long subword units.
 * - Provides methods for encoding, decoding, and saving/loading the tokenizer model.

 * Usage:
 * - Train the tokenizer using the --train-tokenizer option.
 * - Specify the training data path with --data.
 * - Set the vocabulary size with --vocab-size.

 * Dependencies:
 * - Boost libraries for program options and filesystem operations.

 * Author: Cydral
 * Date: 2025, January
 */
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <queue>
#include <queue>
#include <thread>
#include <mutex>
#include <algorithm>
#include <utility>
#include <regex>
#include <locale>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <dlib/serialize.h>
#include <dlib/base64.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// ----------------------------------------------------------------------------------------
// Enumeration defining text processing flags
enum text_processing : uint16_t {
    NONE = 0,
    REPLACE_MULTIPLE_SPACES = 1 << 0, // Replace multiple consecutive spaces with a single space
    REPLACE_MULTIPLE_NEWLINES = 1 << 1, // Replace multiple consecutive newlines with two newlines
    FULL = 0xFFFF // Apply all text processing operations
};

// Replaces multiple consecutive spaces in the input text with a single space
void replace_multiple_spaces(std::string& text) {
    std::regex multiple_spaces(R"(\s{2,})");
    text = std::regex_replace(text, multiple_spaces, " ");
}

// Replaces multiple consecutive newlines in the input text with two newlines
void replace_multiple_newlines(std::string& text) {
    std::regex multiple_newlines(R"(\n{2,})");
    text = std::regex_replace(text, multiple_newlines, "\n");
}

// Applies text preprocessing operations based on the specified flags
void preprocess_text(std::string& text, int16_t processing_flags = text_processing::FULL) {    
    if (processing_flags & REPLACE_MULTIPLE_NEWLINES) replace_multiple_newlines(text);
    if (processing_flags & REPLACE_MULTIPLE_SPACES) replace_multiple_spaces(text);
}

// ----------------------------------------------------------------------------------------
// Function to load data from a file or directory
std::string load_data_from_file_or_directory(const std::string& path, size_t max_size = 0.1 * 1024 * 1024) {
    std::string data;
    size_t total_size = 0;
    bool max_size_reached = false;
    const size_t buffer_size = (4 * 1024);

    auto process_file = [&](const std::string& file_path) {
        std::ifstream input(file_path, std::ios::binary);
        if (input.is_open()) {
            std::cout << "Loading file: " << file_path << std::endl;

            std::vector<char> buffer(buffer_size);
            bool first_chunk = true;

            while (input.read(buffer.data(), buffer_size) || input.gcount() > 0) {
                size_t bytes_read = input.gcount();

                if (!max_size_reached) {
                    size_t remaining_space = max_size - total_size;
                    size_t bytes_to_add = std::min(remaining_space, bytes_read);

                    if (bytes_to_add > 0) {
                        if (!first_chunk && !data.empty()) data += "\n\n";
                        data.append(buffer.data(), bytes_to_add);
                        total_size += bytes_to_add;
                        first_chunk = false;
                    }

                    if (total_size >= max_size) {
                        max_size_reached = true;
                        std::cout << "Max size limit reached. Further content will be ignored." << std::endl;
                        break;
                    }
                }
                else break;  // No need to continue reading if max size is reached
            }
        }
    };

    try {
        if (fs::is_directory(path)) {
            // Recursively traverse the directory
            for (const auto& entry : fs::recursive_directory_iterator(path)) {
                if (fs::is_regular_file(entry)) {
                    if (max_size_reached) break;
                    process_file(entry.path().string());
                }
            }
        }
        else if (fs::is_regular_file(path)) process_file(path);
        else std::cerr << "Path is neither a file nor a directory: " << path << std::endl;
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    std::cout << "Total data size: " << total_size << " bytes" << std::endl;
    return data;
}

// ----------------------------------------------------------------------------------------
// BPE Tokenizer class
class bpe_tokenizer {
public:
    bpe_tokenizer() : vocab_size(BASE_VOCAB_SIZE) {
        // Initialize the base vocabulary with single bytes
        for (int i = 0; i < BASE_VOCAB_SIZE; ++i) {
            vocab[i] = std::vector<uint8_t>{ static_cast<uint8_t>(i) };
        }
        // Initialize special tokens with sequential IDs
        special_tokens = {
            {"<text>",      BASE_VOCAB_SIZE},
            {"</text>",     BASE_VOCAB_SIZE + 1},
            {"<url>",       BASE_VOCAB_SIZE + 2},
            {"</url>",      BASE_VOCAB_SIZE + 3},
            {"<image>",     BASE_VOCAB_SIZE + 4},
            {"</image>",    BASE_VOCAB_SIZE + 5},
            {"<video>",     BASE_VOCAB_SIZE + 6},
            {"</video>",    BASE_VOCAB_SIZE + 7},
            {"<audio>",     BASE_VOCAB_SIZE + 8},
            {"</audio>",    BASE_VOCAB_SIZE + 9},
            {"<file>",      BASE_VOCAB_SIZE + 10},
            {"</file>",     BASE_VOCAB_SIZE + 11},
            {"<code>",      BASE_VOCAB_SIZE + 12},
            {"</code>",     BASE_VOCAB_SIZE + 13},
            {"<summary>",   BASE_VOCAB_SIZE + 14},
            {"</summary>",  BASE_VOCAB_SIZE + 15},
            {"<think>",     BASE_VOCAB_SIZE + 16},
            {"</think>",    BASE_VOCAB_SIZE + 17},
            {"<start>",     BASE_VOCAB_SIZE + 18},
            {"<end>",       BASE_VOCAB_SIZE + 19},
            {"<user>",      BASE_VOCAB_SIZE + 20},
            {"<bot>",       BASE_VOCAB_SIZE + 21},
            {"<system>",    BASE_VOCAB_SIZE + 22},
            {"<question>",  BASE_VOCAB_SIZE + 23},
            {"<answer>",    BASE_VOCAB_SIZE + 24},
            {"<search>",    BASE_VOCAB_SIZE + 25},
            {"<unk>",       BASE_VOCAB_SIZE + 26},
            {"<pad>",       BASE_VOCAB_SIZE + 27}
        };

        // Initialize the vector of special token IDs
        for (const auto& token : special_tokens)
            special_token_map[token.second] = token.first;
    }

    // Train the tokenizer on the given text
    void train(const std::string& text, int vocab_size, bool verbose = false) {
        assert(vocab_size >= BASE_VOCAB_SIZE);
        this->vocab_size = vocab_size;
        int num_merges = vocab_size - BASE_VOCAB_SIZE;

        // Convert text to byte IDs
        std::vector<int> ids;
        for (char c : text) ids.push_back(static_cast<uint8_t>(c));

        // Perform BPE merges
        for (int i = 0; i < num_merges; ++i) {
            auto stats = get_stats(ids);
            if (stats.empty()) break;

            // Find the most frequent pair that does not exceed MAX_TOKEN_LENGTH
            auto pair = get_most_frequent_pair(stats);

            // Check if the resulting token would exceed MAX_TOKEN_LENGTH
            size_t new_token_length = vocab[pair.first].size() + vocab[pair.second].size();
            if (new_token_length > MAX_TOKEN_LENGTH) {
                if (verbose) {
                    std::cout << "\r"
                        << std::setw(100) << std::flush
                        << "\rskipping merge " << std::to_string(i + 1) << "/" << std::to_string(num_merges) << ": ("
                        << std::to_string(pair.first) << "," << std::to_string(pair.second) << ") -> new token length "
                        << std::to_string(new_token_length) << " exceeds limit of " << std::to_string(MAX_TOKEN_LENGTH)
                        << std::flush;
                }
                continue; // Skip this merge
            }

            int idx = (BASE_VOCAB_SIZE + (int)special_tokens.size()) + i;
            ids = merge(ids, pair, idx);
            merges[pair] = idx;
            vocab[idx].insert(vocab[idx].end(), vocab[pair.first].begin(), vocab[pair.first].end());
            vocab[idx].insert(vocab[idx].end(), vocab[pair.second].begin(), vocab[pair.second].end());

            if (verbose) {
                std::cout << "\r"
                    << std::setw(100) << std::flush
                    << "\rmerge " << std::to_string(i + 1) << "/" << std::to_string(num_merges) << ": ("
                    << std::to_string(pair.first) << "," << std::to_string(pair.second) << ") -> " << std::to_string(idx)
                    << " (" << bytes_to_string(vocab[idx]) << ") had "
                    << std::to_string(stats[pair]) << " occurrences"
                    << std::flush;
            }
        }
        std::cout << "\ntraining done\n";
    }

    // Encode the given text into subword tokens
    std::vector<int> encode(const std::string& text) {
        std::vector<int> result_ids;

        // Split the text into paragraphs based on newline characters
        std::vector<std::string> paragraphs;
        size_t start = 0, end = text.find('\n');
        while (end != std::string::npos) {
            // Extract the paragraph and add it only if it's not empty
            std::string paragraph = text.substr(start, end - start + 1);
            if (!paragraph.empty() && paragraph != "\n") paragraphs.push_back(std::move(paragraph));
            start = end + 1;
            end = text.find('\n', start);
        }
        // Add the last paragraph (if any) and only if it's not empty
        if (start < text.size()) {
            std::string paragraph = text.substr(start);
            if (!paragraph.empty() && paragraph != "\n") paragraphs.push_back(std::move(paragraph));
        }

        // Encode each paragraph separately
        int sot_tok = get_special_token_id("<text>");
        int eot_tok = get_special_token_id("</text>");
        for (const auto& paragraph : paragraphs) {
            // Add the <text> token at the beginning of each paragraph
            result_ids.push_back(sot_tok);

            // Convert the paragraph to byte IDs
            std::vector<int> ids;
            ids.reserve(paragraph.size());
            for (char c : paragraph) ids.push_back(static_cast<uint8_t>(c));

            // Precompute valid pairs and their merge order
            auto stats = get_stats(ids);
            std::priority_queue<std::pair<int, std::pair<int, int>>> pq;

            // Initialize the priority queue with valid pairs
            for (const auto& stat : stats) {
                const std::pair<int, int>& pair = stat.first;
                if (merges.count(pair)) {
                    pq.push({ merges.at(pair), pair });
                }
            }

            // Merge pairs in order of their merge priority
            while (!pq.empty()) {
                const auto& top_element = pq.top();
                int merge_order = top_element.first;
                const std::pair<int, int>& pair = top_element.second;
                pq.pop();

                // Check if the pair still exists in the current ids sequence
                bool pair_found = false;
                for (size_t i = 0; i < ids.size() - 1; ++i) {
                    if (ids[i] == pair.first && ids[i + 1] == pair.second) {
                        pair_found = true;
                        break;
                    }
                }
                if (!pair_found) continue;

                // Merge the pair
                int idx = merges.at(pair);
                ids = merge(ids, pair, idx);

                // Update statistics and priority queue with new pairs formed after merging
                stats = get_stats(ids);
                for (const auto& stat : stats) {
                    const std::pair<int, int>& new_pair = stat.first;
                    if (merges.count(new_pair)) {
                        pq.push({ merges.at(new_pair), new_pair });
                    }
                }
            }

            // Append the encoded paragraph to the result
            result_ids.insert(result_ids.end(), ids.begin(), ids.end());

            // Add the </text> token at the end of each paragraph
            result_ids.push_back(eot_tok);
        }

        return result_ids;
    }

    // Decode a single token ID back into text
    std::string decode(int id, bool display_special_tokens = true) {
        return decode(std::vector<int>({ id }), display_special_tokens);
    }

    // Decode a sequence of token IDs back into text
    std::string decode(const std::vector<int>& ids, bool display_special_tokens = true) {
        std::vector<uint8_t> bytes;
        for (int id : ids) {
            // Check if the ID is a special token
            auto it = special_token_map.find(id);
            if (it != special_token_map.end()) {
                // It's a special token, get the corresponding string
                if (display_special_tokens) bytes.insert(bytes.end(), it->second.begin(), it->second.end());
            } else {
                // It's a regular token, get the bytes from the vocabulary
                auto& token = vocab.at(id);
                bytes.insert(bytes.end(), token.begin(), token.end());
            }
        }
        return std::string(bytes.begin(), bytes.end());
    }

    // Save the tokenizer model and vocabulary to file
    friend void serialize(const bpe_tokenizer& tok, std::ostream& out) {
        dlib::serialize("bpe_tokenizer_", out);
        //---
        int nb_merges = tok.merges.size();
        dlib::serialize(nb_merges, out);
        for (int idx = (BASE_VOCAB_SIZE + (int)tok.special_tokens.size()); idx < (tok.vocab_size + tok.special_tokens.size()); ++idx) {
            for (const auto& merge_pair : tok.merges) {
                if (merge_pair.second == idx) {
                    dlib::serialize(merge_pair.first.first, out);
                    dlib::serialize(merge_pair.first.second, out);
                    break;
                }
            }
        }
        //---
        int nb_vocab = (int)tok.vocab.size();
        dlib::serialize(nb_vocab, out);
        for (const auto& v : tok.vocab) {
            std::string token_str = tok.bytes_to_string(v.second);
            dlib::serialize(token_str, out);
            dlib::serialize(v.first, out);
        }
    }

    // Load the tokenizer model and vocabulary from file
    friend void deserialize(bpe_tokenizer& tok, std::istream& in) {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "bpe_tokenizer_")
            throw dlib::serialization_error("Unexpected version '" + version + "' found while deserializing dlib::bpe_tokenizer_.");
        //---
        int idx = BASE_VOCAB_SIZE + (int)tok.special_tokens.size(), nb_merges, nb_vocab, a, b;
        tok.merges.clear();
        dlib::deserialize(nb_merges, in);
        for (int m = 0; m < nb_merges; m++) {
            dlib::deserialize(a, in);
            dlib::deserialize(b, in);
            tok.merges[{a, b}] = idx;
            idx++;
        }
        //---
        std::string token_str;
        tok.vocab.clear();
        dlib::deserialize(nb_vocab, in);
        for (int v = 0; v < nb_vocab; v++) {
            dlib::deserialize(token_str, in);
            dlib::deserialize(idx, in);
            tok.vocab[idx] = tok.string_to_bytes(token_str);
        }
    }

    // Get the ID of a special token
    int get_special_token_id(const std::string& token) const {
        auto it = special_tokens.find(token);
        if (it != special_tokens.end()) return it->second;
        throw std::runtime_error("Special token not found: " + token);
    }

    // Get the total vocabulary size
    size_t get_vocab_size(void) const {
        return (vocab.size() + special_tokens.size());
    }

private:
    std::map<std::string, int> special_tokens;
    std::unordered_map<int, std::string> special_token_map;
    std::map<std::pair<int, int>, int> merges;
    std::map<int, std::vector<uint8_t>> vocab;
    int vocab_size;

    static const size_t MAX_TOKEN_LENGTH = 8;
    static const int BASE_VOCAB_SIZE = 256;

    // Get frequency statistics of adjacent token pairs
    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ (hash2 << 1);
        }
    };
    std::unordered_map<std::pair<int, int>, int, pair_hash> get_stats(const std::vector<int>& ids) {
        std::unordered_map<std::pair<int, int>, int, pair_hash> global_stats;
        std::mutex global_stats_mutex;

        auto worker = [&](size_t start, size_t end) {
            std::unordered_map<std::pair<int, int>, int, pair_hash> local_stats;
            for (size_t i = start; i < end - 1 && i + 1 < ids.size(); ++i)
                local_stats[{ids[i], ids[i + 1]}]++;

            std::lock_guard<std::mutex> lock(global_stats_mutex);
            for (const auto& pair : local_stats)
                global_stats[pair.first] += pair.second;
        };

        size_t num_threads = std::thread::hardware_concurrency();
        size_t segment_size = ids.size() / num_threads;
        std::vector<std::thread> threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * segment_size;
            size_t end = (t == num_threads - 1) ? ids.size() : start + segment_size;
            threads.emplace_back(worker, start, end);
        }

        for (auto& thread : threads) thread.join();

        return global_stats;
    }

    // Finds the most frequent pair of tokens in the given statistics map that does not exceed the maximum token length
    std::pair<int, int> get_most_frequent_pair(const std::unordered_map<std::pair<int, int>, int, pair_hash>& stats) {
        std::pair<int, int> best_pair = { -1, -1 }; // Initialize the best pair to an invalid value
        double max_score = 0; // Initialize the maximum score to 0

        // Iterate over all pairs in the statistics map
        for (const auto& stat : stats) {
            const std::pair<int, int>& pair = stat.first; // Extract the token pair
            int count = stat.second; // Extract the frequency count

            // Check if the new token formed by merging the pair would exceed the maximum allowed length
            size_t new_token_length = vocab[pair.first].size() + vocab[pair.second].size();
            if (new_token_length > MAX_TOKEN_LENGTH) continue; // Skip this pair if it exceeds the maximum token length

            // Calculate the score for this pair (frequency * length_penalty)
            double score = (size_t)count * (new_token_length > (MAX_TOKEN_LENGTH / 2) ? 1.75 : 1.0);

            // Update the best pair if the current pair has a higher score
            if (score > max_score) {
                best_pair = pair;
                max_score = score;
            }
        }

        return best_pair; // Return the pair with the highest score
    }

    // Merge the most frequent pair in the token sequence
    std::vector<int> merge(std::vector<int>& ids, const std::pair<int, int>& pair, int idx) {
        std::vector<int> new_ids;
        new_ids.reserve(ids.size()); // Reserve space to avoid reallocations

        for (size_t i = 0; i < ids.size(); ++i) {
            if (i < ids.size() - 1 && ids[i] == pair.first && ids[i + 1] == pair.second) {
                new_ids.push_back(idx); // Replace the pair with the new token ID
                i++; // Skip the next token
            }
            else new_ids.push_back(ids[i]); // Keep the current token
        }

        return new_ids;
    }

    // Decode/Encode a base64 string to/from a UTF-8 string
    static std::string base64_decode(const std::string& base64_str)
    {
        dlib::base64 decoder;
        std::istringstream sin(base64_str);
        std::ostringstream sout;
        decoder.decode(sin, sout);
        return sout.str();
    }
    static std::string base64_encode(const std::string& input) {
        dlib::base64 encoder;
        std::istringstream sin(input);
        std::ostringstream sout;
        encoder.encode(sin, sout);
        return sout.str();
    }

    // Convert a sequence of bytes to a readable string
    static std::string bytes_to_string(const std::vector<uint8_t>& bytes) {
        std::string data(bytes.begin(), bytes.end());
        return base64_encode(data);
    }

    // Convert a string representation of bytes back to bytes
    static std::vector<uint8_t> string_to_bytes(const std::string& str) {
        std::string decoded = base64_decode(str);
        return std::vector<uint8_t>(decoded.begin(), decoded.end());
    }
};

// Main function to handle command-line arguments and execute the tokenizer
int main(int argc, char* argv[]) {
    std::locale::global(std::locale("en_US.UTF-8"));
    po::options_description desc("Options");
    desc.add_options()
        ("train-tokenizer", "Train the BPE tokenizer")
        ("data", po::value<std::string>(), "Specify a file or directory containing the training data")
        ("data-size", po::value<size_t>()->default_value(10), "Set the maximum size of data to load (in MB, default: 10MB)")
        ("vocab-size", po::value<int>()->default_value(500), "Set the target vocabulary size (default: 500)")
        ("help", "Print this help message");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    if (vm.count("help") || (argc == 1)) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::string data = "The quick brown fox jumps over the lazy dog, and the dog barks loudly!";
    if (vm.count("data")) {
        std::string data_path = vm["data"].as<std::string>();
        size_t data_size = vm["data-size"].as<size_t>() * 1024 * 1024;
        std::cout << "Loading data from: " << data_path << std::endl;
        data = load_data_from_file_or_directory(data_path, data_size);
        if (data.empty()) {
            std::cerr << "Error: No data found in the specified path" << std::endl;
            return 1;
        }
        std::cout << "Data loaded successfully. Size: " << data.size() << " bytes" << std::endl;
        std::cout << "Cleaning text in progress... ";
        preprocess_text(data);
        std::cout << "done\n";
    }

    if (vm.count("train-tokenizer")) {
        bpe_tokenizer tokenizer;
        std::cout << "Training BPE tokenizer on data..." << std::endl;
        int vocab_size = vm["vocab-size"].as<int>();
        tokenizer.train(data, vocab_size, true);

        std::string str_vocab_size = vocab_size >= 1000 ? (std::to_string(vocab_size / 1000) + "k") : std::to_string(vocab_size);
        std::string file_prefix = "dlib_t" + str_vocab_size + "_base";
        dlib::serialize(file_prefix + ".vocab") << tokenizer;
        std::cout << "Model saved to " << file_prefix << ".[model|vocab]" << std::endl;
        dlib::deserialize(file_prefix + ".vocab") >> tokenizer;

        // Test strings in different languages
        std::vector<std::string> test_strings = {
            u8"This is a test of the tokenisation process implemented in the Dlib library!", // English
            u8"Ceci est un test du processus de tokenisation implémenté dans la bibliothèque Dlib!", // French
            u8"Un tamm eo arnod ar prosess tokenadur implijet el levraoueg Dlib!", // Breton
            u8"Dette er en test af tokeniseringsprocessen implementeret i Dlib-biblioteket!", // Danish
            u8"这是对Dlib库中实现的标记化过程的测试！" // Chinese
        };

        for (const auto& test : test_strings) {
            std::cout << "Original: " << test << "\n";

            auto encoded = tokenizer.encode(test);
            std::cout << "Encoded: ";
            for (int id : encoded) std::cout << id << " ";
            std::cout << "\n";
            std::string decoded = tokenizer.decode(encoded, false);
            if (decoded == test) {
                std::cout << "Test passed: decoded string matches the original string!\n";
                // Modify the encoded vector to test special tokens
                if (!encoded.empty()) {
                    encoded[3] = tokenizer.get_special_token_id("<unk>");
                    encoded.push_back(tokenizer.get_special_token_id("<pad>"));
                    encoded.push_back(tokenizer.get_special_token_id("<pad>"));
                    encoded.push_back(tokenizer.get_special_token_id("<pad>"));
                }
                decoded = tokenizer.decode(encoded);
                std::cout << "Decoded with special tokens: " << decoded << "\n";
            } else {
                std::cout << "Test failed: decoded string does not match the original string!\n";
                std::cout << "Decoded string: " << decoded << "\n";
            }
            std::cout << "----------------------------------------\n";
        }
    }

    return 0;
}