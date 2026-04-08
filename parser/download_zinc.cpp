#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>

#if defined(USE_ZLIB_NG)
#include <zlib-ng.h>
#else
#include <zlib.h>
#endif

#if defined(USE_ZLIB_NG)
#define gzopen_compat zng_gzopen
#define gzgets_compat zng_gzgets
#define gzwrite_compat zng_gzwrite
#define gzeof_compat zng_gzeof
#define gzerror_compat zng_gzerror
#define gzclose_compat zng_gzclose
#else
#define gzopen_compat gzopen
#define gzgets_compat gzgets
#define gzwrite_compat gzwrite
#define gzeof_compat gzeof
#define gzerror_compat gzerror
#define gzclose_compat gzclose
#endif

void log_message(const std::string &msg) {
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::cout << msg << std::endl;
}

struct ChunkInput {
  std::string path;
  std::string baseName;
  std::uint32_t chunkId = 0;
};

struct ShortId {
  static constexpr std::size_t kMaxLen = 12;
  std::uint8_t len = 0;
  std::array<char, kMaxLen> data{};

  bool operator==(const ShortId &other) const noexcept {
    return len == other.len &&
           (len == 0 || std::memcmp(data.data(), other.data.data(), len) == 0);
  }
};

namespace std {
template <>
struct hash<ShortId> {
  std::size_t operator()(const ShortId &id) const noexcept {
    uint64_t h = 0;
    if (id.len > 0) {
      std::memcpy(&h, id.data.data(), std::min<std::size_t>(8, id.len));
      if (id.len > 8) {
        uint32_t extra = 0;
        std::memcpy(&extra, id.data.data() + 8, id.len - 8);
        h ^= static_cast<uint64_t>(extra) << 32;
      }
    }
    return h ^ (static_cast<uint64_t>(id.len) << 56);
  }
};
}  // namespace std

struct ProgramOptions {
  std::size_t producers = 1;
  std::string curlScript;
  std::string smiDir = ".";
  bool remainOnlyDeduplicated = false;
};

struct DownloadEntry {
  std::vector<std::string> tokens;
  std::string url;
  std::filesystem::path outputPath;
  ChunkInput chunk;
};

struct ChunkStats {
  std::uint64_t total = 0;
  std::uint64_t unique = 0;
};

struct ProducerStats {
  std::uint64_t total = 0;
  std::uint64_t unique = 0;
};

enum class DownloadResult { kSuccess, kNotFound, kFailed };

bool ends_with(std::string_view value, std::string_view suffix) {
  return suffix.size() <= value.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

std::string to_lower_copy(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  for (unsigned char c : in) {
    out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

std::string trim_copy(std::string_view in) {
  while (!in.empty() && std::isspace(static_cast<unsigned char>(in.front()))) {
    in.remove_prefix(1);
  }
  while (!in.empty() && std::isspace(static_cast<unsigned char>(in.back()))) {
    in.remove_suffix(1);
  }
  return std::string(in);
}

bool parse_positive_size(const std::string &value, std::size_t &out) {
  try {
    std::size_t idx = 0;
    unsigned long long parsed = std::stoull(value, &idx, 10);
    if (idx != value.size() || parsed == 0) {
      return false;
    }
    out = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " --curl-script <file> [--smi-dir <dir>] [--producers <n>]\n"
            << "       [--remain_only_deduplicated]\n"
            << "       (writes chunk_table.csv into --smi-dir)\n";
}

bool parse_arguments(int argc, char *argv[], ProgramOptions &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--curl-script") {
      if (i + 1 >= argc) {
        std::cerr << "--curl-script requires a path\n";
        return false;
      }
      opts.curlScript = argv[++i];
    } else if (arg == "--smi-dir") {
      if (i + 1 >= argc) {
        std::cerr << "--smi-dir requires a path\n";
        return false;
      }
      opts.smiDir = argv[++i];
    } else if (arg == "--producers" || arg == "-p") {
      if (i + 1 >= argc || !parse_positive_size(argv[i + 1], opts.producers)) {
        std::cerr << "Invalid producer count\n";
        return false;
      }
      ++i;
    } else if (arg == "--remain_only_deduplicated") {
      opts.remainOnlyDeduplicated = true;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }
  if (opts.curlScript.empty()) {
    std::cerr << "--curl-script is required\n";
    return false;
  }
  if (opts.producers == 0) {
    opts.producers = 1;
  }
  return true;
}

bool ensure_directory(const std::filesystem::path &p, const char *desc) {
  std::error_code ec;
  if (std::filesystem::exists(p, ec)) {
    if (!std::filesystem::is_directory(p, ec)) {
      std::cerr << desc << " is not a directory: " << p << "\n";
      return false;
    }
    return true;
  }
  if (std::filesystem::create_directories(p, ec)) {
    return true;
  }
  std::cerr << "Failed to create " << desc << ": " << p;
  if (ec) {
    std::cerr << " (" << ec.message() << ")";
  }
  std::cerr << "\n";
  return false;
}

bool ensure_parent_directory(const std::filesystem::path &p) {
  const std::filesystem::path parent = p.parent_path();
  if (parent.empty()) {
    return true;
  }
  return ensure_directory(parent, "download directory");
}

bool has_zinc_prefix(std::string_view id) {
  if (id.size() < 4) {
    return false;
  }
  const auto to_lower = [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };
  return to_lower(id[0]) == 'z' && to_lower(id[1]) == 'i' &&
         to_lower(id[2]) == 'n' && to_lower(id[3]) == 'c';
}

bool make_short_id(std::string_view idView, ShortId &out) {
  if (has_zinc_prefix(idView)) {
    idView.remove_prefix(4);
  }
  if (idView.empty() || idView.size() > ShortId::kMaxLen) {
    return false;
  }
  out.len = static_cast<std::uint8_t>(idView.size());
  std::memcpy(out.data.data(), idView.data(), idView.size());
  if (idView.size() < ShortId::kMaxLen) {
    std::memset(out.data.data() + idView.size(), 0,
                ShortId::kMaxLen - idView.size());
  }
  return true;
}

bool tokenize_shell_command(const std::string &line,
                            std::vector<std::string> &tokens) {
  tokens.clear();
  std::string current;
  bool inSingle = false;
  bool inDouble = false;
  bool escape = false;
  for (char ch : line) {
    if (escape) {
      current.push_back(ch);
      escape = false;
      continue;
    }
    if (inSingle) {
      if (ch == '\'') {
        inSingle = false;
      } else {
        current.push_back(ch);
      }
      continue;
    }
    if (inDouble) {
      if (ch == '\"') {
        inDouble = false;
      } else if (ch == '\\') {
        escape = true;
      } else {
        current.push_back(ch);
      }
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(ch))) {
      if (!current.empty()) {
        tokens.push_back(current);
        current.clear();
      }
      continue;
    }
    if (ch == '\\') {
      escape = true;
      continue;
    }
    if (ch == '\'') {
      inSingle = true;
      continue;
    }
    if (ch == '\"') {
      inDouble = true;
      continue;
    }
    current.push_back(ch);
  }
  if (escape || inSingle || inDouble) {
    return false;
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return !tokens.empty();
}

bool extract_curl_output(const std::vector<std::string> &tokens,
                         std::filesystem::path &outPath) {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const std::string &tok = tokens[i];
    if (tok == "-o" || tok == "--output") {
      if (i + 1 >= tokens.size()) {
        return false;
      }
      outPath = tokens[i + 1];
      return true;
    }
    if (tok.size() > 2 && tok.rfind("-o", 0) == 0) {
      outPath = tok.substr(2);
      return true;
    }
  }
  return false;
}

bool extract_curl_url(const std::vector<std::string> &tokens,
                      std::string &url) {
  for (const auto &tok : tokens) {
    if (tok.rfind("https://", 0) == 0 || tok.rfind("http://", 0) == 0) {
      url = tok;
      return true;
    }
  }
  return false;
}

bool parse_curl_line(const std::string &line,
                     std::size_t lineNumber,
                     DownloadEntry &entry) {
  std::vector<std::string> tokens;
  if (!tokenize_shell_command(line, tokens)) {
    std::cerr << "Failed to parse line " << lineNumber
              << " (unbalanced quotes or escapes)\n";
    return false;
  }
  if (tokens.empty()) {
    return false;
  }
  const std::filesystem::path cmd(tokens[0]);
  if (cmd.filename() != "curl") {
    std::cerr << "Line " << lineNumber << " does not start with curl\n";
    return false;
  }

  std::filesystem::path outputPath;
  if (!extract_curl_output(tokens, outputPath)) {
    std::cerr << "Line " << lineNumber << " is missing -o/--output\n";
    return false;
  }
  std::string url;
  if (!extract_curl_url(tokens, url)) {
    std::cerr << "Line " << lineNumber << " is missing a URL\n";
    return false;
  }

  entry.tokens = std::move(tokens);
  entry.url = std::move(url);
  entry.outputPath = outputPath;
  entry.chunk.path = outputPath.string();
  entry.chunk.baseName = outputPath.stem().string();
  return true;
}

bool read_curl_script(const std::string &scriptPath,
                      std::vector<DownloadEntry> &out) {
  std::ifstream input(scriptPath);
  if (!input) {
    std::cerr << "Failed to open curl script: " << scriptPath << "\n";
    return false;
  }

  out.clear();
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    const std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    DownloadEntry entry;
    if (!parse_curl_line(trimmed, lineNumber, entry)) {
      return false;
    }
    out.emplace_back(std::move(entry));
  }

  if (out.empty()) {
    std::cerr << "Curl script did not contain any entries: " << scriptPath
              << "\n";
    return false;
  }
  return true;
}

std::string make_chunk_prefix(const ChunkInput &chunk) {
  std::ostringstream oss;
  if (!chunk.baseName.empty()) {
    oss << chunk.baseName << '_';
  }
  oss << "chunk_" << chunk.chunkId;
  return oss.str();
}

std::filesystem::path make_chunk_output_path(const ChunkInput &chunk,
                                             const std::filesystem::path &outDir) {
  return outDir / (make_chunk_prefix(chunk) + ".gz");
}

bool write_chunk_table(const std::filesystem::path &outDir,
                       const std::vector<DownloadEntry> &entries) {
  const std::filesystem::path tablePath = outDir / "chunk_table.csv";
  std::ofstream out(tablePath, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to write chunk table: " << tablePath << "\n";
    return false;
  }
  out << "chunk_id,abs_path,rel_path\n";
  std::size_t written = 0;
  std::error_code ec;
  for (const auto &entry : entries) {
    const auto outPath = make_chunk_output_path(entry.chunk, outDir);
    if (!std::filesystem::exists(outPath)) {
      continue;
    }
    std::filesystem::path absPath = std::filesystem::absolute(outPath, ec);
    if (ec) {
      absPath = outPath;
      ec.clear();
    }
    const std::string relPath = outPath.filename().string();
    out << entry.chunk.chunkId << "," << absPath.string() << "," << relPath << "\n";
    ++written;
  }
  if (written == 0) {
    std::cerr << "No chunk outputs found to write in " << outDir << "\n";
    return false;
  }
  log_message("Wrote chunk table: " + tablePath.string() +
              " (rows: " + std::to_string(written) + ")");
  return true;
}

bool process_chunk(const ChunkInput &chunk,
                   const std::filesystem::path &outDir,
                   ChunkStats &stats) {
  log_message("Filtering duplicate ZINC IDs: " + chunk.path);
  stats = {};
  const std::filesystem::path gzPath =
      make_chunk_output_path(chunk, outDir);

  gzFile gzOut = gzopen_compat(gzPath.string().c_str(), "wb");
  if (!gzOut) {
    std::cerr << "Failed to open gzip for writing: " << gzPath << "\n";
    return false;
  }
  auto cleanupGzip = [&]() {
    if (gzOut) {
      gzclose_compat(gzOut);
      gzOut = nullptr;
    }
    std::error_code ec;
    std::filesystem::remove(gzPath, ec);
  };

  bool hasLastId = false;
  ShortId lastId{};
  std::uint64_t totalCount = 0;
  std::uint64_t uniqueCount = 0;
  std::string serialized;
  serialized.reserve(256);

  auto handleLine = [&](const std::string &rawLine) -> bool {
    std::string_view view(rawLine);
    while (!view.empty() &&
           (view.back() == '\n' || view.back() == '\r')) {
      view.remove_suffix(1);
    }
    if (view.empty()) {
      return true;
    }
    const std::size_t firstTab = view.find('\t');
    if (firstTab == std::string::npos) {
      return true;
    }
    const std::size_t secondTab = view.find('\t', firstTab + 1);
    const std::string_view smilesView = view.substr(0, firstTab);
    const std::string_view idView =
        (secondTab == std::string_view::npos)
            ? view.substr(firstTab + 1)
            : view.substr(firstTab + 1, secondTab - firstTab - 1);
    if (smilesView.empty() || idView.empty()) {
      return true;
    }

    ShortId idKey;
    if (!make_short_id(idView, idKey)) {
      return true;
    }
    ++totalCount;
    if (hasLastId && idKey == lastId) {
      return true;
    }
    lastId = idKey;
    hasLastId = true;

    serialized.clear();
    serialized.append(smilesView);
    serialized.push_back('\t');
    serialized.append(idView);
    serialized.push_back('\n');

    const ssize_t written =
        gzwrite_compat(gzOut, serialized.data(),
                       static_cast<unsigned int>(serialized.size()));
    if (written < 0 ||
        static_cast<std::size_t>(written) != serialized.size()) {
      std::cerr << "Failed to write gzip chunk: " << gzPath << "\n";
      return false;
    }

    ++uniqueCount;
    return true;
  };

  const bool isGz = ends_with(to_lower_copy(chunk.path), ".gz");
  if (!isGz) {
    std::ifstream input(chunk.path);
    if (!input) {
      std::cerr << "Failed to open input file: " << chunk.path << "\n";
      cleanupGzip();
      return false;
    }
    std::string line;
    while (std::getline(input, line)) {
      if (!handleLine(line)) {
        cleanupGzip();
        return false;
      }
    }
  } else {
    gzFile file = gzopen_compat(chunk.path.c_str(), "rb");
    if (!file) {
      std::cerr << "Failed to open gzip input file: " << chunk.path << "\n";
      cleanupGzip();
      return false;
    }
    auto closeFile = [&]() { gzclose_compat(file); };
    std::string line;
    constexpr int bufferSize = 8192;
    std::array<char, bufferSize> buffer{};
    bool ok = true;
    while (ok) {
      line.clear();
      while (true) {
        char *res = gzgets_compat(file, buffer.data(), bufferSize);
        if (!res) {
          if (gzeof_compat(file)) {
            break;
          }
          int errNum = 0;
          const char *errMsg = gzerror_compat(file, &errNum);
          std::cerr << "Error reading gzip file: " << chunk.path << " ("
                    << (errMsg ? errMsg : "unknown") << ")\n";
          ok = false;
          break;
        }
        line.append(res);
        if (!line.empty() && line.back() == '\n') {
          break;
        }
      }
      if (!ok) {
        break;
      }
      if (!line.empty()) {
        if (!handleLine(line)) {
          ok = false;
          break;
        }
      }
      if (gzeof_compat(file)) {
        break;
      }
    }
    closeFile();
    if (!ok) {
      cleanupGzip();
      return false;
    }
  }

  if (gzclose_compat(gzOut) != Z_OK) {
    std::cerr << "Failed to close gzip file: " << gzPath << "\n";
    gzOut = nullptr;
    std::error_code ec;
    std::filesystem::remove(gzPath, ec);
    return false;
  }
  gzOut = nullptr;

  stats.total = totalCount;
  stats.unique = uniqueCount;
  log_message("Finished chunk " + std::to_string(chunk.chunkId) +
              " total molecules: " + std::to_string(totalCount) +
              " kept after ID filter: " + std::to_string(uniqueCount));
  return true;
}

DownloadResult run_curl_command(const DownloadEntry &entry) {
  if (!ensure_parent_directory(entry.outputPath)) {
    return DownloadResult::kFailed;
  }
  log_message("Downloading: " + entry.url + " -> " +
              entry.outputPath.string());

  std::vector<char *> argv;
  argv.reserve(entry.tokens.size() + 1);
  for (const auto &token : entry.tokens) {
    argv.push_back(const_cast<char *>(token.c_str()));
  }
  argv.push_back(nullptr);

  pid_t pid = fork();
  if (pid == -1) {
    std::cerr << "Failed to fork for curl: " << std::strerror(errno) << "\n";
    return DownloadResult::kFailed;
  }
  if (pid == 0) {
    execvp(argv[0], argv.data());
    std::cerr << "Failed to exec curl: " << std::strerror(errno) << "\n";
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) == -1) {
    std::cerr << "Failed to wait for curl: " << std::strerror(errno) << "\n";
    return DownloadResult::kFailed;
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    if (!std::filesystem::exists(entry.outputPath)) {
      std::cerr << "curl reported success but output is missing: "
                << entry.outputPath << "\n";
      return DownloadResult::kFailed;
    }
    return DownloadResult::kSuccess;
  }
  if (WIFEXITED(status)) {
    const int code = WEXITSTATUS(status);
    if (code == 22) {
      std::cerr << "curl returned 404 for " << entry.url << ", skipping\n";
      std::error_code ec;
      std::filesystem::remove(entry.outputPath, ec);
      return DownloadResult::kNotFound;
    }
    std::cerr << "curl exited with status " << code << "\n";
  } else if (WIFSIGNALED(status)) {
    std::cerr << "curl terminated by signal " << WTERMSIG(status) << "\n";
  }
  return DownloadResult::kFailed;
}

void producer_thread(const std::vector<DownloadEntry> &entries,
                     std::atomic<std::size_t> &nextIndex,
                     std::atomic<bool> &hadError,
                     const std::filesystem::path &outDir,
                     bool remainOnlyDeduplicated,
                     ProducerStats &stats) {
  while (!hadError.load()) {
    const std::size_t idx = nextIndex.fetch_add(1);
    if (idx >= entries.size()) {
      break;
    }
    const DownloadEntry &entry = entries[idx];
    const DownloadResult result = run_curl_command(entry);
    if (result == DownloadResult::kFailed) {
      hadError = true;
      break;
    }
    if (result == DownloadResult::kNotFound) {
      continue;
    }
    ChunkStats chunkStats;
    if (!process_chunk(entry.chunk, outDir, chunkStats)) {
      hadError = true;
      break;
    }
    stats.total += chunkStats.total;
    stats.unique += chunkStats.unique;
    if (remainOnlyDeduplicated) {
      std::error_code ec;
      std::filesystem::remove(entry.outputPath, ec);
      if (ec) {
        std::cerr << "Failed to remove original input: "
                  << entry.outputPath << " (" << ec.message() << ")\n";
      }
    }
  }
}

int main(int argc, char *argv[]) {
  ProgramOptions opts;
  if (!parse_arguments(argc, argv, opts)) {
    print_usage(argv[0]);
    return 1;
  }

  if (!ensure_directory(opts.smiDir, "smi-dir")) {
    return 1;
  }

  std::vector<DownloadEntry> entries;
  if (!read_curl_script(opts.curlScript, entries)) {
    return 1;
  }

  for (std::size_t i = 0; i < entries.size(); ++i) {
    entries[i].chunk.chunkId = static_cast<std::uint32_t>(i);
  }

  const std::size_t producerCount = std::max<std::size_t>(1, opts.producers);
  std::atomic<std::size_t> nextIndex{0};
  std::atomic<bool> hadError{false};
  std::vector<ProducerStats> producerStats(producerCount);
  std::vector<std::thread> producers;
  producers.reserve(producerCount);
  for (std::size_t i = 0; i < producerCount; ++i) {
    producers.emplace_back([&entries, &nextIndex, &hadError, &opts,
                            &producerStats, i]() {
      producer_thread(entries, nextIndex, hadError, opts.smiDir,
                      opts.remainOnlyDeduplicated, producerStats[i]);
    });
  }
  for (auto &thread : producers) {
    thread.join();
  }

  std::uint64_t totalMolecules = 0;
  std::uint64_t totalUnique = 0;
  for (const auto &stats : producerStats) {
    totalMolecules += stats.total;
    totalUnique += stats.unique;
  }
  log_message("Overall molecules: " + std::to_string(totalMolecules) +
              " overall kept after ID filter: " + std::to_string(totalUnique));

  const bool wroteTable = write_chunk_table(opts.smiDir, entries);

  return (hadError || !wroteTable) ? 1 : 0;
}
