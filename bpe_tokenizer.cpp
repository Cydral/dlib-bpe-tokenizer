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
 * - Handles special tokens such as <|endoftext|>, <|unk|>, and <|pad|>.
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
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Function to load data from a file or directory
std::string load_data_from_file_or_directory(const std::string& path, size_t max_size = 0.1 * 1024 * 1024) {
    std::string data;
    size_t total_size = 0;
    bool max_size_reached = false;
    const size_t buffer_size = 4 * 1024;

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
                else {
                    break;  // No need to continue reading if max size is reached
                }
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
        else if (fs::is_regular_file(path)) {
            process_file(path);
        }
        else {
            std::cerr << "Path is neither a file nor a directory: " << path << std::endl;
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    std::cout << "Total data size: " << total_size << " bytes" << std::endl;
    return data;
}

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
            {"<|startoftext|>", BASE_VOCAB_SIZE},
            {"<|endoftext|>", BASE_VOCAB_SIZE + 1},
            {"<|question|>", BASE_VOCAB_SIZE + 2},
            {"<|response|>", BASE_VOCAB_SIZE + 3},
            {"<|unk|>", (BASE_VOCAB_SIZE + 4)},
            {"<|pad|>", (BASE_VOCAB_SIZE + 5)}
        };
    }

    // Train the tokenizer on the given text
    void train(const std::string& text, int vocab_size, bool verbose = false) {
        assert(vocab_size >= BASE_VOCAB_SIZE);
        this->vocab_size = vocab_size;
        int num_merges = vocab_size - BASE_VOCAB_SIZE;

        // Convert text to byte IDs
        std::vector<int> ids;
        for (char c : text) {
            ids.push_back(static_cast<uint8_t>(c));
        }

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
        // Convert text to byte IDs
        std::vector<int> ids;
        ids.reserve(text.size()); // Reserve space to avoid reallocations
        for (char c : text) {
            ids.push_back(static_cast<uint8_t>(c));
        }

        // Precompute valid pairs and their merge order
        auto stats = get_stats(ids); // Compute initial statistics
        std::priority_queue<std::pair<int, std::pair<int, int>>> pq; // Min-heap based on merge order

        // Initialize the priority queue with valid pairs
        for (const auto& stat : stats) {
            const std::pair<int, int>& pair = stat.first;
            if (merges.count(pair)) {
                pq.push({ merges.at(pair), pair }); // Use merge order as the key
            }
        }

        // Merge pairs in order of their merge priority
        while (!pq.empty()) {
            const auto& top_element = pq.top(); // Get the pair with the smallest merge order
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
            if (!pair_found) continue; // Skip if the pair no longer exists

            // Merge the pair
            int idx = merges.at(pair);
            ids = merge(ids, pair, idx); // Use an optimized merge function

            // Update statistics and priority queue with new pairs formed after merging
            stats = get_stats(ids);
            for (const auto& stat : stats) {
                const std::pair<int, int>& new_pair = stat.first;
                if (merges.count(new_pair)) {
                    pq.push({ merges.at(new_pair), new_pair });
                }
            }
        }

        return ids;
    }

    // Decode a single token ID back into text
    std::string decode(int id) {
        for (const auto& token : special_tokens) {
            if (token.second == id) {
                return token.first;
            }
        }
        auto& token = vocab.at(id);
        return std::string(token.begin(), token.end());
    }

    // Decode a sequence of token IDs back into text
    std::string decode(const std::vector<int>& ids) {
        std::vector<uint8_t> bytes;
        for (int id : ids) {
            bool is_special_token = false;
            for (const auto& token : special_tokens) {
                if (token.second == id) {
                    bytes.insert(bytes.end(), token.first.begin(), token.first.end());
                    is_special_token = true;
                    break;
                }
            }
            if (!is_special_token) {
                auto& token = vocab.at(id);
                bytes.insert(bytes.end(), token.begin(), token.end());
            }
        }
        return std::string(bytes.begin(), bytes.end());
    }

    // Save the tokenizer model and vocabulary to files
    void save(const std::string& file_prefix) {
        std::ofstream model_file(file_prefix + ".model");
        model_file << "bpe-tokenizer v1\n";

        for (int idx = BASE_VOCAB_SIZE + (int)special_tokens.size(); idx < vocab_size; ++idx) {
            for (const auto& merge_pair : merges) {
                if (merge_pair.second == idx) {
                    model_file << merge_pair.first.first << " " << merge_pair.first.second << "\n";
                    break;
                }
            }
        }

        std::ofstream vocab_file(file_prefix + ".vocab");
        for (const auto& v : vocab) {
            vocab_file << "[" << bytes_to_string(v.second) << "] " << v.first << "\n";
        }
    }

    // Load the tokenizer model and vocabulary from files
    bool load(const std::string& model_file) {                
        std::ifstream file(model_file + ".model");
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open model file: " << model_file + ".model" << "\n";
            return false;
        }
        std::string line;
        std::getline(file, line); // Version

        int idx = BASE_VOCAB_SIZE + (int)special_tokens.size(), a, b;
        merges.clear();
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            iss >> a >> b;
            merges[{a, b}] = idx;
            idx++;
        }

        std::ifstream vocab_file(model_file + ".vocab");
        if (!vocab_file.is_open()) {
            std::cerr << "Error: Unable to open vocab file: " << model_file + ".vocab" << "\n";
            return false;
        }
        vocab.clear();
        while (std::getline(vocab_file, line)) {
            // Find the first '[' and the last ']' in the line
            size_t start = line.find('[');
            size_t end = line.rfind(']');  // Use rfind to find the last ']'
            if (start != std::string::npos && end != std::string::npos) {
                std::string token_str = line.substr(start + 1, end - start - 1);
                try {
                    idx = std::stoi(line.substr(end + 2));
                    vocab[idx] = string_to_bytes(token_str);
                }
                catch (const std::invalid_argument& /* e */) {
                    std::cerr << "Error: Invalid token ID in vocab file: " << line << "\n";
                    continue;
                }
            }
        }
        return true;
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
            for (size_t i = start; i < end - 1; ++i) {
                local_stats[{ids[i], ids[i + 1]}]++;
            }

            std::lock_guard<std::mutex> lock(global_stats_mutex);
            for (const auto& pair : local_stats) {
                global_stats[pair.first] += pair.second;
            }
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
        int max_count = 0; // Initialize the maximum frequency count to 0

        // Iterate over all pairs in the statistics map
        for (const auto& stat : stats) {
            const std::pair<int, int>& pair = stat.first; // Extract the token pair
            int count = stat.second; // Extract the frequency count

            // Check if the new token formed by merging the pair would exceed the maximum allowed length
            size_t new_token_length = vocab[pair.first].size() + vocab[pair.second].size();
            if (new_token_length > MAX_TOKEN_LENGTH) {
                continue; // Skip this pair if it exceeds the maximum token length
            }

            // Update the best pair if the current pair has a higher frequency
            if (count > max_count) {
                best_pair = pair;
                max_count = count;
            }
        }

        return best_pair; // Return the most frequent valid pair
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

    // Convert a sequence of bytes to a readable string
    std::string bytes_to_string(const std::vector<uint8_t>& bytes) {
        std::string result;
        for (uint8_t byte : bytes) {
            if (byte >= 32 && byte <= 126) {
                result += static_cast<char>(byte);
            }
            else {
                char buf[5];
                snprintf(buf, sizeof(buf), "\\x%02x", byte);
                result += buf;
            }
        }
        return result;
    }

    // Convert a string representation of bytes back to bytes
    std::vector<uint8_t> string_to_bytes(const std::string& str) {
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < str.length(); ++i) {
            if (str[i] == '\\' && i + 3 < str.length() && str[i + 1] == 'x') {
                char hex[3] = { str[i + 2], str[i + 3], '\0' };
                uint8_t byte = static_cast<uint8_t>(std::stoul(hex, nullptr, 16));
                bytes.push_back(byte);
                i += 3;
            }
            else {
                bytes.push_back(static_cast<uint8_t>(str[i]));
            }
        }
        return bytes;
    }
};

// Main function to handle command-line arguments and execute the tokenizer
int main(int argc, char* argv[]) {
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
    }

    if (vm.count("train-tokenizer")) {
        bpe_tokenizer tokenizer;
        std::string test = "Agroecological research investigates comprehensive environmental interaction mechanisms.";
        std::cout << "Training BPE tokenizer on data..." << std::endl;
        int vocab_size = vm["vocab-size"].as<int>();
        tokenizer.train(data, vocab_size, true);

        std::string str_vocab_size = vocab_size >= 1000 ? (std::to_string(vocab_size / 1000) + "k") : std::to_string(vocab_size);
        std::string file_prefix = "dlib_t" + str_vocab_size + "_base";
        tokenizer.save(file_prefix);
        std::cout << "Model saved to " << file_prefix << ".[model|vocab]" << std::endl;
        tokenizer.load(file_prefix);

        auto encoded = tokenizer.encode(test);
        std::cout << "Encoded: ";
        for (int id : encoded) {
            std::cout << id << " ";
        }
        std::cout << "\n";

        encoded[3] = tokenizer.get_special_token_id("<|unk|>");
        encoded.push_back(tokenizer.get_special_token_id("<|endoftext|>"));
        encoded.push_back(tokenizer.get_special_token_id("<|pad|>"));
        std::string decoded = tokenizer.decode(encoded);
        std::cout << "Decoded: " << decoded << "\n";
    }

    return 0;
}