#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <bzlib.h>
#include <htslib/bgzf.h>
#include <htslib/kstring.h>
#include <zstd.h>

#if defined(USE_ZLIB_NG)
#include <zlib-ng.h>
#else
#include <zlib.h>
#endif

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#define bgzf_open_compat bgzf_open
#define bgzf_close_compat bgzf_close

#if defined(USE_ZLIB_NG)
#define gzopen_compat zng_gzopen
#define gzgets_compat zng_gzgets
#define gzeof_compat zng_gzeof
#define gzerror_compat zng_gzerror
#define gzclose_compat zng_gzclose
#else
#define gzopen_compat gzopen
#define gzgets_compat gzgets
#define gzeof_compat gzeof
#define gzerror_compat gzerror
#define gzclose_compat gzclose
#endif

enum class InputFormat {
  kAuto,
  kSmi,
  kCsv,
  kTsv,
};

enum class HeaderMode {
  kAuto,
  kYes,
  kNo,
};

struct ColumnSpec {
  bool explicitValue = false;
  bool disabled = false;
  bool isIndex = false;
  std::size_t index = 0;
  std::string name;
};

struct ProgramOptions {
  std::size_t producers = std::max(1u, std::thread::hardware_concurrency());
  std::size_t innerThreads = 1;
  std::string chunkTable;
  std::string outDir = ".";
  InputFormat inputFormat = InputFormat::kAuto;
  HeaderMode headerMode = HeaderMode::kAuto;
  ColumnSpec smilesColumn{false, false, false, 0, "smiles"};
  ColumnSpec idColumn{false, false, false, 0, "id"};
  std::string generatedIdPrefix = "mol";
};

struct ChunkInput {
  std::string path;
  std::string relPath;
  std::uint32_t chunkId = 0;
};

void log_message(const std::string &msg) {
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::cout << msg << std::endl;
}

bool parse_positive_size(const std::string &value, std::size_t &out) {
  try {
    std::size_t idx = 0;
    const unsigned long long parsed = std::stoull(value, &idx, 10);
    if (idx != value.size() || parsed == 0) {
      return false;
    }
    out = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_non_negative_size(const std::string &value, std::size_t &out) {
  try {
    std::size_t idx = 0;
    const unsigned long long parsed = std::stoull(value, &idx, 10);
    if (idx != value.size()) {
      return false;
    }
    out = static_cast<std::size_t>(parsed);
    return true;
  } catch (...) {
    return false;
  }
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

void strip_utf8_bom(std::string &line) {
  if (line.size() >= 3 &&
      static_cast<unsigned char>(line[0]) == 0xEF &&
      static_cast<unsigned char>(line[1]) == 0xBB &&
      static_cast<unsigned char>(line[2]) == 0xBF) {
    line.erase(0, 3);
  }
}

std::string normalize_absolute_path(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path normalized = std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    ec.clear();
    normalized = std::filesystem::absolute(path, ec);
  }
  if (ec) {
    return path.string();
  }
  return normalized.string();
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return suffix.size() <= value.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

bool parse_input_format(std::string_view value, InputFormat &out) {
  const std::string lowered = to_lower_copy(trim_copy(value));
  if (lowered == "auto") {
    out = InputFormat::kAuto;
    return true;
  }
  if (lowered == "smi" || lowered == "smiles" || lowered == "cxsmiles" ||
      lowered == "csxmiles") {
    out = InputFormat::kSmi;
    return true;
  }
  if (lowered == "csv") {
    out = InputFormat::kCsv;
    return true;
  }
  if (lowered == "tsv" || lowered == "tab") {
    out = InputFormat::kTsv;
    return true;
  }
  return false;
}

bool parse_header_mode(std::string_view value, HeaderMode &out) {
  const std::string lowered = to_lower_copy(trim_copy(value));
  if (lowered == "auto") {
    out = HeaderMode::kAuto;
    return true;
  }
  if (lowered == "yes" || lowered == "true" || lowered == "1") {
    out = HeaderMode::kYes;
    return true;
  }
  if (lowered == "no" || lowered == "false" || lowered == "0") {
    out = HeaderMode::kNo;
    return true;
  }
  return false;
}

bool parse_column_spec(std::string_view value, ColumnSpec &out) {
  const std::string trimmed = trim_copy(value);
  if (trimmed.empty()) {
    return false;
  }

  out.explicitValue = true;
  out.disabled = false;
  out.isIndex = false;
  out.index = 0;
  out.name.clear();

  const std::string lowered = to_lower_copy(trimmed);
  if (lowered == "none" || lowered == "-") {
    out.disabled = true;
    return true;
  }

  std::size_t index = 0;
  if (parse_non_negative_size(trimmed, index)) {
    out.isIndex = true;
    out.index = index;
    return true;
  }

  out.name = lowered;
  return true;
}

void print_usage(const char *prog) {
  std::cerr
      << "Usage: " << prog
      << " --chunk-table <csv> [--out-dir <dir>] [--format auto|smi|csv|tsv]\n"
         "       [--header auto|yes|no] [--smiles-column <name|index>]\n"
         "       [--id-column <name|index|none>] [--generated-id-prefix <prefix>]\n"
         "       [--producers <n>] [--inner-threads <k>]\n";
}

bool parse_arguments(int argc, char *argv[], ProgramOptions &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--chunk-table") {
      if (i + 1 >= argc) {
        std::cerr << "--chunk-table requires a path\n";
        return false;
      }
      opts.chunkTable = argv[++i];
    } else if (arg == "--out-dir" || arg == "--smi-dir") {
      if (i + 1 >= argc) {
        std::cerr << arg << " requires a path\n";
        return false;
      }
      opts.outDir = argv[++i];
    } else if (arg == "--format") {
      if (i + 1 >= argc || !parse_input_format(argv[i + 1], opts.inputFormat)) {
        std::cerr << "Invalid input format\n";
        return false;
      }
      ++i;
    } else if (arg == "--header") {
      if (i + 1 >= argc || !parse_header_mode(argv[i + 1], opts.headerMode)) {
        std::cerr << "Invalid header mode\n";
        return false;
      }
      ++i;
    } else if (arg == "--smiles-column") {
      if (i + 1 >= argc || !parse_column_spec(argv[i + 1], opts.smilesColumn) ||
          opts.smilesColumn.disabled) {
        std::cerr << "Invalid smiles column selector\n";
        return false;
      }
      ++i;
    } else if (arg == "--id-column") {
      if (i + 1 >= argc || !parse_column_spec(argv[i + 1], opts.idColumn)) {
        std::cerr << "Invalid id column selector\n";
        return false;
      }
      ++i;
    } else if (arg == "--generated-id-prefix") {
      if (i + 1 >= argc) {
        std::cerr << "--generated-id-prefix requires a value\n";
        return false;
      }
      opts.generatedIdPrefix = argv[++i];
    } else if (arg == "--producers" || arg == "-p") {
      if (i + 1 >= argc || !parse_positive_size(argv[i + 1], opts.producers)) {
        std::cerr << "Invalid producer count\n";
        return false;
      }
      ++i;
    } else if (arg == "--inner-threads") {
      if (i + 1 >= argc || !parse_positive_size(argv[i + 1], opts.innerThreads)) {
        std::cerr << "Invalid inner thread count\n";
        return false;
      }
      ++i;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }

  if (opts.chunkTable.empty()) {
    std::cerr << "--chunk-table is required\n";
    return false;
  }
  if (opts.generatedIdPrefix.empty()) {
    std::cerr << "--generated-id-prefix must not be empty\n";
    return false;
  }
  if (opts.producers == 0) {
    opts.producers = 1;
  }
  if (opts.innerThreads == 0) {
    opts.innerThreads = 1;
  }
  return true;
}

bool ensure_directory(const std::filesystem::path &p, const char *desc) {
  if (p.empty()) {
    return true;
  }
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

bool read_chunk_table(const std::string &tablePath, std::vector<ChunkInput> &out) {
  std::ifstream input(tablePath);
  if (!input) {
    std::cerr << "Failed to open chunk table: " << tablePath << "\n";
    return false;
  }

  out.clear();
  std::string line;
  bool firstLine = true;
  while (std::getline(input, line)) {
    const std::string trimmedLine = trim_copy(line);
    if (trimmedLine.empty()) {
      continue;
    }
    if (firstLine) {
      const std::string lowered = to_lower_copy(trimmedLine);
      if (lowered.find("chunk_id") != std::string::npos) {
        firstLine = false;
        continue;
      }
    }
    firstLine = false;

    std::stringstream ss(trimmedLine);
    std::string chunkIdStr;
    std::string absPathStr;
    std::string relPathStr;
    if (!std::getline(ss, chunkIdStr, ',')) {
      continue;
    }
    if (!std::getline(ss, absPathStr, ',')) {
      continue;
    }
    std::getline(ss, relPathStr);

    chunkIdStr = trim_copy(chunkIdStr);
    absPathStr = trim_copy(absPathStr);
    relPathStr = trim_copy(relPathStr);
    if (chunkIdStr.empty() || absPathStr.empty()) {
      continue;
    }

    unsigned long long parsedId = 0;
    try {
      parsedId = std::stoull(chunkIdStr);
    } catch (...) {
      std::cerr << "Invalid chunk id in table: " << chunkIdStr << "\n";
      return false;
    }
    if (parsedId > std::numeric_limits<std::uint32_t>::max()) {
      std::cerr << "Chunk id too large in table: " << chunkIdStr << "\n";
      return false;
    }

    std::filesystem::path absPath(absPathStr);
    if (!std::filesystem::exists(absPath)) {
      std::cerr << "Chunk path from table does not exist: " << absPath << "\n";
      return false;
    }

    ChunkInput chunk;
    chunk.chunkId = static_cast<std::uint32_t>(parsedId);
    chunk.path = normalize_absolute_path(absPath);
    chunk.relPath = relPathStr;
    if (chunk.relPath.empty()) {
      chunk.relPath = absPath.filename().string();
    }
    out.emplace_back(std::move(chunk));
  }

  if (out.empty()) {
    std::cerr << "Chunk table did not contain any entries: " << tablePath << "\n";
    return false;
  }
  return true;
}

std::filesystem::path make_output_relative_path(const ChunkInput &chunk) {
  std::filesystem::path rel(chunk.relPath);
  if (rel.empty()) {
    rel = std::filesystem::path(chunk.path).filename();
  }
  if (rel.is_absolute()) {
    rel = rel.filename();
  }
  const std::string lowered = to_lower_copy(rel.filename().string());
  if (ends_with(lowered, ".bgzf") || ends_with(lowered, ".gz") ||
      ends_with(lowered, ".bz2")) {
    rel.replace_extension(".zst");
  } else if (!ends_with(lowered, ".zst")) {
    rel += ".zst";
  }
  return rel;
}

std::filesystem::path make_output_path(const ChunkInput &chunk,
                                       const std::filesystem::path &outDir) {
  return outDir / make_output_relative_path(chunk);
}

bool write_chunk_table(const std::filesystem::path &outDir,
                       const std::vector<ChunkInput> &chunks) {
  const std::filesystem::path tablePath = outDir / "chunk_table.csv";
  std::ofstream out(tablePath, std::ios::out | std::ios::trunc);
  if (!out) {
    std::cerr << "Failed to write chunk table: " << tablePath << "\n";
    return false;
  }

  out << "chunk_id,abs_path,rel_path\n";
  std::size_t written = 0;
  for (const auto &chunk : chunks) {
    const std::filesystem::path relPath = make_output_relative_path(chunk);
    const std::filesystem::path outPath = outDir / relPath;
    if (!std::filesystem::exists(outPath)) {
      continue;
    }

    out << chunk.chunkId << "," << normalize_absolute_path(outPath) << ","
        << relPath.string() << "\n";
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

enum class LineReadStatus {
  kLine,
  kEof,
  kError,
};

class ChunkLineReader {
 public:
  enum class Kind {
    kNone,
    kPlain,
    kGzip,
    kBgzf,
    kBzip2,
    kZstd,
  };

  ~ChunkLineReader() { close(); }

  bool open(const std::filesystem::path &inputPath) {
    close();
    path_ = inputPath;
    const std::string lowered = to_lower_copy(inputPath.string());

    if (ends_with(lowered, ".bgzf")) {
      bgzf_ = bgzf_open_compat(inputPath.c_str(), "r");
      if (!bgzf_) {
        std::cerr << "Failed to open BGZF input file: " << inputPath << "\n";
        return false;
      }
      kind_ = Kind::kBgzf;
      return true;
    }

    if (ends_with(lowered, ".gz")) {
      gzip_ = gzopen_compat(inputPath.c_str(), "rb");
      if (!gzip_) {
        std::cerr << "Failed to open gzip input file: " << inputPath << "\n";
        return false;
      }
      kind_ = Kind::kGzip;
      return true;
    }

    if (ends_with(lowered, ".bz2")) {
      bzipFile_ = std::fopen(inputPath.c_str(), "rb");
      if (!bzipFile_) {
        std::cerr << "Failed to open bzip2 input file: " << inputPath << "\n";
        return false;
      }
      int bzError = BZ_OK;
      bzip_ = BZ2_bzReadOpen(&bzError, bzipFile_, 0, 0, nullptr, 0);
      if (!bzip_ || bzError != BZ_OK) {
        std::cerr << "Failed to initialize bzip2 reader: " << inputPath << "\n";
        close();
        return false;
      }
      bzipReachedEof_ = false;
      bzipPending_.clear();
      kind_ = Kind::kBzip2;
      return true;
    }

    if (ends_with(lowered, ".zst")) {
      zstdFile_.open(inputPath, std::ios::in | std::ios::binary);
      if (!zstdFile_) {
        std::cerr << "Failed to open ZSTD input file: " << inputPath << "\n";
        return false;
      }
      zstdStream_ = ZSTD_createDStream();
      if (!zstdStream_) {
        std::cerr << "Failed to create ZSTD stream for: " << inputPath << "\n";
        close();
        return false;
      }
      const std::size_t initRet = ZSTD_initDStream(zstdStream_);
      if (ZSTD_isError(initRet)) {
        std::cerr << "Failed to initialize ZSTD stream for: " << inputPath
                  << " (" << ZSTD_getErrorName(initRet) << ")\n";
        close();
        return false;
      }
      zstdInBuf_.resize(ZSTD_DStreamInSize());
      zstdOutBuf_.resize(ZSTD_DStreamOutSize());
      zstdInput_ = {nullptr, 0, 0};
      zstdPending_.clear();
      zstdReachedEof_ = false;
      kind_ = Kind::kZstd;
      return true;
    }

    plain_.open(inputPath, std::ios::in | std::ios::binary);
    if (!plain_) {
      std::cerr << "Failed to open input file: " << inputPath << "\n";
      return false;
    }
    kind_ = Kind::kPlain;
    return true;
  }

  LineReadStatus next_line(std::string &line) {
    line.clear();
    switch (kind_) {
      case Kind::kPlain: {
        if (std::getline(plain_, line)) {
          trim_line_end(line);
          return LineReadStatus::kLine;
        }
        if (plain_.eof()) {
          return LineReadStatus::kEof;
        }
        std::cerr << "Error reading input: " << path_ << "\n";
        return LineReadStatus::kError;
      }
      case Kind::kGzip: {
        while (true) {
          char *res = gzgets_compat(gzip_, gzipBuffer_.data(),
                                    static_cast<int>(gzipBuffer_.size()));
          if (!res) {
            if (gzeof_compat(gzip_)) {
              if (line.empty()) {
                return LineReadStatus::kEof;
              }
              trim_line_end(line);
              return LineReadStatus::kLine;
            }
            int errNum = 0;
            const char *errMsg = gzerror_compat(gzip_, &errNum);
            std::cerr << "Error reading gzip input: " << path_ << " ("
                      << (errMsg ? errMsg : "unknown") << ")\n";
            return LineReadStatus::kError;
          }
          line.append(res);
          if (!line.empty() && line.back() == '\n') {
            trim_line_end(line);
            return LineReadStatus::kLine;
          }
        }
      }
      case Kind::kBgzf: {
        const int ret = bgzf_getline(bgzf_, '\n', &bgzfBuffer_);
        if (ret >= 0) {
          line.assign(bgzfBuffer_.s, bgzfBuffer_.l);
          trim_line_end(line);
          return LineReadStatus::kLine;
        }
        if (ret == -1) {
          return LineReadStatus::kEof;
        }
        std::cerr << "Error reading BGZF input: " << path_ << "\n";
        return LineReadStatus::kError;
      }
      case Kind::kBzip2: {
        while (true) {
          const std::size_t newlinePos = bzipPending_.find('\n');
          if (newlinePos != std::string::npos) {
            line.assign(bzipPending_.data(), newlinePos);
            bzipPending_.erase(0, newlinePos + 1);
            trim_line_end(line);
            return LineReadStatus::kLine;
          }

          if (bzipReachedEof_) {
            if (bzipPending_.empty()) {
              return LineReadStatus::kEof;
            }
            line.swap(bzipPending_);
            trim_line_end(line);
            return LineReadStatus::kLine;
          }

          int bzError = BZ_OK;
          const int bytesRead = BZ2_bzRead(&bzError, bzip_, bzipBuffer_.data(),
                                           static_cast<int>(bzipBuffer_.size()));
          if (bzError != BZ_OK && bzError != BZ_STREAM_END) {
            std::cerr << "Error reading bzip2 input: " << path_ << "\n";
            return LineReadStatus::kError;
          }
          if (bytesRead > 0) {
            bzipPending_.append(bzipBuffer_.data(),
                                static_cast<std::size_t>(bytesRead));
          }
          if (bzError == BZ_STREAM_END) {
            bzipReachedEof_ = true;
          }
        }
      }
      case Kind::kZstd: {
        while (true) {
          const std::size_t newlinePos = zstdPending_.find('\n');
          if (newlinePos != std::string::npos) {
            line.assign(zstdPending_.data(), newlinePos);
            zstdPending_.erase(0, newlinePos + 1);
            trim_line_end(line);
            return LineReadStatus::kLine;
          }

          if (zstdReachedEof_) {
            if (zstdPending_.empty()) {
              return LineReadStatus::kEof;
            }
            line.swap(zstdPending_);
            trim_line_end(line);
            return LineReadStatus::kLine;
          }

          if (!fill_zstd_pending()) {
            return LineReadStatus::kError;
          }
        }
      }
      case Kind::kNone:
        break;
    }
    return LineReadStatus::kError;
  }

  void close() {
    if (bgzf_) {
      bgzf_close_compat(bgzf_);
      bgzf_ = nullptr;
    }
    if (gzip_) {
      gzclose_compat(gzip_);
      gzip_ = nullptr;
    }
    if (bzip_) {
      int bzError = BZ_OK;
      BZ2_bzReadClose(&bzError, bzip_);
      bzip_ = nullptr;
    }
    if (bzipFile_) {
      std::fclose(bzipFile_);
      bzipFile_ = nullptr;
    }
    if (zstdStream_) {
      ZSTD_freeDStream(zstdStream_);
      zstdStream_ = nullptr;
    }
    if (plain_.is_open()) {
      plain_.close();
    }
    if (zstdFile_.is_open()) {
      zstdFile_.close();
    }
    if (bgzfBuffer_.s) {
      std::free(bgzfBuffer_.s);
      bgzfBuffer_.s = nullptr;
      bgzfBuffer_.l = bgzfBuffer_.m = 0;
    }
    bzipPending_.clear();
    bzipReachedEof_ = false;
    zstdPending_.clear();
    zstdReachedEof_ = false;
    zstdInput_ = {nullptr, 0, 0};
    zstdInBuf_.clear();
    zstdOutBuf_.clear();
    kind_ = Kind::kNone;
    path_.clear();
  }

 private:
  bool fill_zstd_pending() {
    while (true) {
      if (zstdInput_.pos == zstdInput_.size) {
        zstdFile_.read(zstdInBuf_.data(),
                       static_cast<std::streamsize>(zstdInBuf_.size()));
        const std::streamsize got = zstdFile_.gcount();
        if (got < 0 || zstdFile_.bad()) {
          std::cerr << "Error reading ZSTD input: " << path_ << "\n";
          return false;
        }
        zstdInput_.src = zstdInBuf_.data();
        zstdInput_.size = static_cast<std::size_t>(got);
        zstdInput_.pos = 0;
      }

      ZSTD_outBuffer output{zstdOutBuf_.data(), zstdOutBuf_.size(), 0};
      const std::size_t ret =
          ZSTD_decompressStream(zstdStream_, &output, &zstdInput_);
      if (ZSTD_isError(ret)) {
        std::cerr << "ZSTD decompression error in " << path_ << ": "
                  << ZSTD_getErrorName(ret) << "\n";
        return false;
      }
      if (output.pos > 0) {
        zstdPending_.append(zstdOutBuf_.data(), output.pos);
        return true;
      }
      if (zstdInput_.size == 0 && zstdInput_.pos == 0) {
        zstdReachedEof_ = true;
        return true;
      }
    }
  }

  static void trim_line_end(std::string &line) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
  }

  Kind kind_ = Kind::kNone;
  std::ifstream plain_;
  gzFile gzip_ = nullptr;
  BGZF *bgzf_ = nullptr;
  kstring_t bgzfBuffer_{0, 0, nullptr};
  std::array<char, 8192> gzipBuffer_{};
  std::FILE *bzipFile_ = nullptr;
  BZFILE *bzip_ = nullptr;
  bool bzipReachedEof_ = false;
  std::array<char, 64 * 1024> bzipBuffer_{};
  std::string bzipPending_;
  std::ifstream zstdFile_;
  ZSTD_DStream *zstdStream_ = nullptr;
  std::vector<char> zstdInBuf_;
  std::vector<char> zstdOutBuf_;
  ZSTD_inBuffer zstdInput_{nullptr, 0, 0};
  std::string zstdPending_;
  bool zstdReachedEof_ = false;
  std::filesystem::path path_;
};

bool write_zstd_frame(ZSTD_CStream *cstream, std::ofstream &outFile,
                      const std::string &buffer) {
  ZSTD_inBuffer input = {buffer.data(), buffer.size(), 0};
  while (input.pos < input.size) {
    char outBuff[128 * 1024];
    ZSTD_outBuffer output = {outBuff, sizeof(outBuff), 0};
    const size_t ret = ZSTD_compressStream(cstream, &output, &input);
    if (ZSTD_isError(ret)) {
      std::cerr << "ZSTD compression error: " << ZSTD_getErrorName(ret) << "\n";
      return false;
    }
    outFile.write(outBuff, static_cast<std::streamsize>(output.pos));
  }

  ZSTD_outBuffer output = {nullptr, 0, 0};
  size_t ret = ZSTD_endStream(cstream, &output);
  if (ret != 0) {
    char outBuff[128 * 1024];
    ZSTD_outBuffer output2 = {outBuff, sizeof(outBuff), 0};
    do {
      ret = ZSTD_endStream(cstream, &output2);
      if (ZSTD_isError(ret)) {
        std::cerr << "ZSTD compression error: " << ZSTD_getErrorName(ret) << "\n";
        return false;
      }
      outFile.write(outBuff, static_cast<std::streamsize>(output2.pos));
      output2.pos = 0;
    } while (ret > 0);
  }
  return true;
}

bool parse_delimited_line(std::string_view line, char delimiter,
                          std::vector<std::string> &fields) {
  fields.clear();
  std::string field;
  field.reserve(line.size());
  bool inQuotes = false;

  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (inQuotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          inQuotes = false;
        }
      } else {
        field.push_back(c);
      }
      continue;
    }

    if (c == delimiter) {
      fields.emplace_back(std::move(field));
      field.clear();
      continue;
    }
    if (c == '"') {
      if (!field.empty()) {
        return false;
      }
      inQuotes = true;
      continue;
    }
    field.push_back(c);
  }

  if (inQuotes) {
    return false;
  }
  fields.emplace_back(std::move(field));
  return true;
}

bool find_column(const std::vector<std::string> &headerFields,
                 std::string_view targetName, std::size_t &columnIndex) {
  const std::string loweredTarget = to_lower_copy(targetName);
  for (std::size_t i = 0; i < headerFields.size(); ++i) {
    if (to_lower_copy(trim_copy(headerFields[i])) == loweredTarget) {
      columnIndex = i;
      return true;
    }
  }
  return false;
}

InputFormat guess_input_format(const std::filesystem::path &path) {
  const std::string lowered = to_lower_copy(path.filename().string());
  if (ends_with(lowered, ".csv") || ends_with(lowered, ".csv.gz") ||
      ends_with(lowered, ".csv.bz2") || ends_with(lowered, ".csv.zst")) {
    return InputFormat::kCsv;
  }
  if (ends_with(lowered, ".tsv") || ends_with(lowered, ".tsv.gz") ||
      ends_with(lowered, ".tsv.bz2") || ends_with(lowered, ".tsv.zst") ||
      ends_with(lowered, ".tab") || ends_with(lowered, ".tab.gz") ||
      ends_with(lowered, ".tab.bz2") || ends_with(lowered, ".tab.zst")) {
    return InputFormat::kTsv;
  }
  return InputFormat::kSmi;
}

bool should_treat_first_row_as_header(const std::vector<std::string> &fields,
                                      const ProgramOptions &opts) {
  if (opts.smilesColumn.isIndex) {
    return false;
  }

  std::size_t smilesColumn = 0;
  const std::string smilesName =
      opts.smilesColumn.name.empty() ? "smiles" : opts.smilesColumn.name;
  if (!find_column(fields, smilesName, smilesColumn)) {
    return false;
  }

  if (opts.idColumn.disabled || opts.idColumn.isIndex || !opts.idColumn.explicitValue) {
    return true;
  }

  std::size_t idColumn = 0;
  return find_column(fields, opts.idColumn.name, idColumn);
}

bool resolve_delimited_columns(const std::vector<std::string> &fields,
                               const ProgramOptions &opts, bool treatAsHeader,
                               std::size_t &smilesColumn, bool &hasIdColumn,
                               std::size_t &idColumn) {
  const std::size_t fieldCount = fields.size();

  if (treatAsHeader) {
    if (opts.smilesColumn.isIndex) {
      smilesColumn = opts.smilesColumn.index;
      if (smilesColumn >= fieldCount) {
        std::cerr << "SMILES column index " << smilesColumn
                  << " is out of range for the header\n";
        return false;
      }
    } else {
      const std::string smilesName =
          opts.smilesColumn.name.empty() ? "smiles" : opts.smilesColumn.name;
      if (!find_column(fields, smilesName, smilesColumn)) {
        std::cerr << "Could not find SMILES column named '" << smilesName
                  << "' in the header\n";
        return false;
      }
    }

    if (opts.idColumn.disabled) {
      hasIdColumn = false;
      return true;
    }

    if (opts.idColumn.isIndex) {
      if (opts.idColumn.index < fieldCount) {
        hasIdColumn = true;
        idColumn = opts.idColumn.index;
      } else if (opts.idColumn.explicitValue) {
        std::cerr << "ID column index " << opts.idColumn.index
                  << " is out of range for the header\n";
        return false;
      } else {
        hasIdColumn = false;
      }
      return true;
    }

    const std::string idName = opts.idColumn.name.empty() ? "id" : opts.idColumn.name;
    if (find_column(fields, idName, idColumn)) {
      hasIdColumn = true;
      return true;
    }
    if (opts.idColumn.explicitValue) {
      std::cerr << "Could not find ID column named '" << idName
                << "' in the header\n";
      return false;
    }
    hasIdColumn = false;
    return true;
  }

  if (opts.smilesColumn.isIndex) {
    smilesColumn = opts.smilesColumn.index;
  } else if (opts.smilesColumn.explicitValue) {
    std::cerr << "Named SMILES columns require a header row\n";
    return false;
  } else {
    smilesColumn = 0;
  }
  if (smilesColumn >= fieldCount) {
    std::cerr << "SMILES column index " << smilesColumn
              << " is out of range for a data row\n";
    return false;
  }

  if (opts.idColumn.disabled) {
    hasIdColumn = false;
    return true;
  }

  if (opts.idColumn.isIndex) {
    if (opts.idColumn.index < fieldCount) {
      hasIdColumn = true;
      idColumn = opts.idColumn.index;
    } else if (opts.idColumn.explicitValue) {
      std::cerr << "ID column index " << opts.idColumn.index
                << " is out of range for a data row\n";
      return false;
    } else {
      hasIdColumn = false;
    }
    return true;
  }

  if (opts.idColumn.explicitValue) {
    std::cerr << "Named ID columns require a header row\n";
    return false;
  }

  if (fieldCount > 1) {
    hasIdColumn = true;
    idColumn = 1;
  } else {
    hasIdColumn = false;
  }
  return true;
}

struct ResolvedLayout {
  enum class Kind {
    kSmi,
    kDelimited,
  };

  Kind kind = Kind::kSmi;
  char delimiter = ',';
  std::size_t smilesColumn = 0;
  bool hasIdColumn = false;
  std::size_t idColumn = 0;
  std::string firstDataLine;
  std::size_t firstDataLineNum = 0;
};

bool read_next_nonempty_line(ChunkLineReader &reader, std::string &line,
                             std::size_t &lineNum, bool &eofReached) {
  eofReached = false;
  while (true) {
    const LineReadStatus status = reader.next_line(line);
    if (status == LineReadStatus::kError) {
      return false;
    }
    if (status == LineReadStatus::kEof) {
      eofReached = true;
      return true;
    }
    ++lineNum;
    if (trim_copy(line).empty()) {
      continue;
    }
    return true;
  }
}

bool initialize_layout(ChunkLineReader &reader, const ChunkInput &chunk,
                       const ProgramOptions &opts, ResolvedLayout &layout,
                       std::size_t &lineNum) {
  std::string firstLine;
  bool eofReached = false;
  if (!read_next_nonempty_line(reader, firstLine, lineNum, eofReached)) {
    return false;
  }
  if (eofReached) {
    std::cerr << "Chunk is empty: " << chunk.path << "\n";
    return false;
  }

  strip_utf8_bom(firstLine);
  const InputFormat format =
      opts.inputFormat == InputFormat::kAuto ? guess_input_format(chunk.path)
                                             : opts.inputFormat;

  if (format == InputFormat::kSmi) {
    layout.kind = ResolvedLayout::Kind::kSmi;
    layout.firstDataLine = std::move(firstLine);
    layout.firstDataLineNum = lineNum;
    return true;
  }

  layout.kind = ResolvedLayout::Kind::kDelimited;
  layout.delimiter = format == InputFormat::kTsv ? '\t' : ',';

  std::vector<std::string> fields;
  if (!parse_delimited_line(firstLine, layout.delimiter, fields)) {
    std::cerr << "Failed to parse first row in delimited chunk: " << chunk.path << "\n";
    return false;
  }

  bool treatAsHeader = false;
  if (opts.headerMode == HeaderMode::kYes) {
    treatAsHeader = true;
  } else if (opts.headerMode == HeaderMode::kAuto) {
    treatAsHeader = should_treat_first_row_as_header(fields, opts);
  }

  if (!resolve_delimited_columns(fields, opts, treatAsHeader, layout.smilesColumn,
                                 layout.hasIdColumn, layout.idColumn)) {
    return false;
  }

  if (!treatAsHeader) {
    layout.firstDataLine = std::move(firstLine);
    layout.firstDataLineNum = lineNum;
  }
  return true;
}

struct RawLine {
  std::string line;
  std::size_t sourceLineNum = 0;
};

struct ProcessedLine {
  std::string outLine;
  bool valid = false;
};

std::string make_generated_id(const std::string &prefix, std::uint32_t chunkId,
                              std::size_t lineNum) {
  std::string id;
  id.reserve(prefix.size() + 32);
  id.append(prefix);
  id.push_back('_');
  id.append(std::to_string(chunkId));
  id.push_back('_');
  id.append(std::to_string(lineNum));
  return id;
}

void normalize_identifier(std::string &value) {
  value = trim_copy(value);
  for (char &c : value) {
    if (c == '\t' || c == '\r' || c == '\n') {
      c = ' ';
    }
  }
}

void worker_routine(const std::vector<RawLine> &inputs,
                    std::vector<ProcessedLine> &outputs, std::size_t start,
                    std::size_t end, const ResolvedLayout &layout,
                    const std::string &generatedIdPrefix,
                    std::uint32_t chunkId) {
  std::vector<std::string> fields;
  for (std::size_t i = start; i < end; ++i) {
    auto &output = outputs[i];
    output.valid = false;
    output.outLine.clear();

    std::string id;
    RDKit::ROMol *rawMol = nullptr;

    if (layout.kind == ResolvedLayout::Kind::kSmi) {
      const std::string line = trim_copy(inputs[i].line);
      if (line.empty()) {
        continue;
      }
      try {
        rawMol = RDKit::SmilesToMol(line);
      } catch (...) {
      }
      std::unique_ptr<RDKit::ROMol> mol(rawMol);
      if (!mol) {
        continue;
      }

      if (!mol->getPropIfPresent("_Name", id) || trim_copy(id).empty()) {
        id = make_generated_id(generatedIdPrefix, chunkId, inputs[i].sourceLineNum);
      }
      normalize_identifier(id);

      try {
        RDKit::MolOps::removeStereochemistry(*mol);
        const std::string normalizedSmiles = RDKit::MolToSmiles(*mol, false);
        output.outLine.reserve(normalizedSmiles.size() + id.size() + 2);
        output.outLine.append(normalizedSmiles);
        output.outLine.push_back('\t');
        output.outLine.append(id);
        output.outLine.push_back('\n');
        output.valid = true;
      } catch (...) {
      }
      continue;
    }

    if (!parse_delimited_line(inputs[i].line, layout.delimiter, fields)) {
      continue;
    }
    if (layout.smilesColumn >= fields.size()) {
      continue;
    }

    const std::string smiles = trim_copy(fields[layout.smilesColumn]);
    if (smiles.empty()) {
      continue;
    }

    if (layout.hasIdColumn && layout.idColumn < fields.size()) {
      id = trim_copy(fields[layout.idColumn]);
    }
    if (id.empty()) {
      id = make_generated_id(generatedIdPrefix, chunkId, inputs[i].sourceLineNum);
    }
    normalize_identifier(id);

    try {
      rawMol = RDKit::SmilesToMol(smiles);
    } catch (...) {
    }
    std::unique_ptr<RDKit::ROMol> mol(rawMol);
    if (!mol) {
      continue;
    }

    try {
      RDKit::MolOps::removeStereochemistry(*mol);
      const std::string normalizedSmiles = RDKit::MolToSmiles(*mol, false);
      output.outLine.reserve(normalizedSmiles.size() + id.size() + 2);
      output.outLine.append(normalizedSmiles);
      output.outLine.push_back('\t');
      output.outLine.append(id);
      output.outLine.push_back('\n');
      output.valid = true;
    } catch (...) {
    }
  }
}

bool process_chunk(const ChunkInput &chunk, const std::filesystem::path &outDir,
                   const ProgramOptions &opts) {
  log_message("Normalizing chunk: " + chunk.path);
  const std::filesystem::path outPath = make_output_path(chunk, outDir);
  if (!ensure_directory(outPath.parent_path(), "output directory")) {
    return false;
  }

  std::error_code ec;
  const std::filesystem::path inAbs = std::filesystem::absolute(chunk.path, ec);
  const std::filesystem::path outAbs = std::filesystem::absolute(outPath, ec);
  if (!inAbs.empty() && !outAbs.empty() && inAbs == outAbs) {
    std::cerr << "Output path matches input path: " << outPath << "\n";
    return false;
  }

  ChunkLineReader reader;
  if (!reader.open(chunk.path)) {
    return false;
  }

  ResolvedLayout layout;
  std::size_t lineNum = 0;
  if (!initialize_layout(reader, chunk, opts, layout, lineNum)) {
    reader.close();
    return false;
  }

  std::ofstream outFile(outPath, std::ios::binary);
  if (!outFile) {
    std::cerr << "Failed to open output file: " << outPath << "\n";
    reader.close();
    return false;
  }

  ZSTD_CStream *cstream = ZSTD_createCStream();
  if (!cstream) {
    std::cerr << "Failed to create ZSTD stream\n";
    reader.close();
    return false;
  }
  if (ZSTD_isError(ZSTD_initCStream(cstream, 19))) {
    std::cerr << "Failed to initialize ZSTD stream\n";
    ZSTD_freeCStream(cstream);
    reader.close();
    return false;
  }

  std::string outputBuffer;
  outputBuffer.reserve(524288);

  std::size_t readLines = layout.firstDataLine.empty() ? 0 : 1;
  std::size_t processed = 0;
  std::size_t written = 0;
  std::size_t skipped = 0;
  bool ok = true;
  bool firstDataPending = !layout.firstDataLine.empty();

  const std::size_t batchSize = 200000;
  std::vector<RawLine> rawBatch;
  rawBatch.reserve(batchSize);
  std::vector<ProcessedLine> processedBatch;
  processedBatch.reserve(batchSize);

  auto cleanup = [&]() {
    reader.close();
    if (cstream) {
      ZSTD_freeCStream(cstream);
      cstream = nullptr;
    }
  };

  while (true) {
    rawBatch.clear();
    bool eof = false;
    std::string lineBuffer;

    while (rawBatch.size() < batchSize) {
      if (firstDataPending) {
        RawLine raw;
        raw.line = layout.firstDataLine;
        raw.sourceLineNum = layout.firstDataLineNum;
        rawBatch.emplace_back(std::move(raw));
        firstDataPending = false;
        continue;
      }

      const LineReadStatus status = reader.next_line(lineBuffer);
      if (status != LineReadStatus::kLine) {
        if (status == LineReadStatus::kEof) {
          eof = true;
        }
        if (status == LineReadStatus::kError) {
          ok = false;
        }
        break;
      }

      ++lineNum;
      if (trim_copy(lineBuffer).empty()) {
        continue;
      }

      ++readLines;
      RawLine raw;
      raw.line = std::move(lineBuffer);
      raw.sourceLineNum = lineNum;
      rawBatch.emplace_back(std::move(raw));
      lineBuffer.clear();
    }

    if (!ok) {
      break;
    }
    if (rawBatch.empty()) {
      if (eof) {
        break;
      }
      continue;
    }

    processed += rawBatch.size();
    processedBatch.resize(rawBatch.size());

    if (opts.innerThreads <= 1) {
      worker_routine(rawBatch, processedBatch, 0, rawBatch.size(), layout,
                     opts.generatedIdPrefix, chunk.chunkId);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(opts.innerThreads);
      const std::size_t chunkSize =
          (rawBatch.size() + opts.innerThreads - 1) / opts.innerThreads;
      for (std::size_t t = 0; t < opts.innerThreads; ++t) {
        const std::size_t start = t * chunkSize;
        const std::size_t end = std::min(start + chunkSize, rawBatch.size());
        if (start < end) {
          workers.emplace_back(worker_routine, std::cref(rawBatch),
                               std::ref(processedBatch), start, end,
                               std::cref(layout), std::cref(opts.generatedIdPrefix),
                               chunk.chunkId);
        }
      }
      for (auto &worker : workers) {
        worker.join();
      }
    }

    for (const auto &res : processedBatch) {
      if (!res.valid) {
        ++skipped;
        continue;
      }

      outputBuffer.append(res.outLine);
      ++written;

      if (outputBuffer.size() >= 8 * 524288) {
        if (!write_zstd_frame(cstream, outFile, outputBuffer)) {
          std::cerr << "Failed to write ZSTD output: " << outPath << "\n";
          ok = false;
          break;
        }
        outputBuffer.clear();
      }
    }
    if (!ok) {
      break;
    }

    static std::atomic<std::size_t> batchCountGlobal = 0;
    if (++batchCountGlobal % 5 == 0) {
      log_message("Chunk " + std::to_string(chunk.chunkId) + " read " +
                  std::to_string(readLines) + " lines so far...");
    }

    if (eof) {
      break;
    }
  }

  if (ok && !outputBuffer.empty()) {
    if (!write_zstd_frame(cstream, outFile, outputBuffer)) {
      std::cerr << "Failed to write ZSTD output: " << outPath << "\n";
      ok = false;
    }
  }

  cleanup();
  if (!ok) {
    std::error_code rmec;
    std::filesystem::remove(outPath, rmec);
    return false;
  }

  log_message("Finished chunk " + std::to_string(chunk.chunkId) +
              " read: " + std::to_string(readLines) +
              " processed: " + std::to_string(processed) +
              " written: " + std::to_string(written) +
              " skipped (invalid rows): " + std::to_string(skipped));
  return true;
}

void producer_thread(const std::vector<ChunkInput> &inputs,
                     std::atomic<std::size_t> &nextIndex,
                     std::atomic<bool> &hadError,
                     const std::filesystem::path &outDir,
                     const ProgramOptions &opts) {
  while (true) {
    const std::size_t idx = nextIndex.fetch_add(1);
    if (idx >= inputs.size()) {
      break;
    }
    if (!process_chunk(inputs[idx], outDir, opts)) {
      hadError.store(true, std::memory_order_relaxed);
      log_message("Failed to process: " + inputs[idx].path);
    }
  }
}

int main(int argc, char *argv[]) {
  ProgramOptions opts;
  if (!parse_arguments(argc, argv, opts)) {
    print_usage(argv[0]);
    return 1;
  }

  std::vector<ChunkInput> chunks;
  if (!read_chunk_table(opts.chunkTable, chunks)) {
    return 1;
  }
  if (!ensure_directory(opts.outDir, "out-dir")) {
    return 1;
  }

  const std::size_t producerCount = std::max<std::size_t>(1, opts.producers);
  std::atomic<std::size_t> nextIndex{0};
  std::atomic<bool> hadError{false};
  std::vector<std::thread> producers;
  producers.reserve(producerCount);
  for (std::size_t i = 0; i < producerCount; ++i) {
    producers.emplace_back([&chunks, &nextIndex, &hadError, &opts]() {
      producer_thread(chunks, nextIndex, hadError, opts.outDir, opts);
    });
  }
  for (auto &thread : producers) {
    thread.join();
  }

  const bool wroteTable = write_chunk_table(opts.outDir, chunks);
  return (hadError || !wroteTable) ? 1 : 0;
}
