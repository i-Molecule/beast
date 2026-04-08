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
#include <zstd.h>

#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

struct ProgramOptions {
  std::size_t producers = std::max(1u, std::thread::hardware_concurrency());
  std::size_t innerThreads = 1;
  std::string chunkTable;
  std::string outDir = ".";
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

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " --chunk-table <csv> [--out-dir <dir>] [--producers <n>] [--inner-threads <k>]\n";
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

struct ChunkLineReader {
  enum class Kind {
    kNone,
    kPlain,
    kBzip2,
  };

  Kind kind = Kind::kNone;
  std::ifstream plain;
  std::FILE *bzipFile = nullptr;
  BZFILE *bzip = nullptr;
  bool bzipReachedEof = false;
  std::array<char, 64 * 1024> bzipBuffer{};
  std::string bzipPending;
  std::filesystem::path path;

  ~ChunkLineReader() { close(); }

  bool open(const std::filesystem::path &inputPath) {
    close();
    path = inputPath;
    const std::string lowered = to_lower_copy(inputPath.string());
    if (ends_with(lowered, ".bz2")) {
      bzipFile = std::fopen(inputPath.c_str(), "rb");
      if (!bzipFile) {
        std::cerr << "Failed to open bzip2 input file: " << inputPath << "\n";
        return false;
      }
      int bzError = BZ_OK;
      bzip = BZ2_bzReadOpen(&bzError, bzipFile, 0, 0, nullptr, 0);
      if (!bzip || bzError != BZ_OK) {
        std::cerr << "Failed to initialize bzip2 reader: " << inputPath << "\n";
        close();
        return false;
      }
      bzipReachedEof = false;
      bzipPending.clear();
      kind = Kind::kBzip2;
      return true;
    }

    plain.open(inputPath, std::ios::in | std::ios::binary);
    if (!plain) {
      std::cerr << "Failed to open input file: " << inputPath << "\n";
      return false;
    }
    kind = Kind::kPlain;
    return true;
  }

  LineReadStatus next_line(std::string &line) {
    line.clear();
    switch (kind) {
      case Kind::kPlain: {
        if (std::getline(plain, line)) {
          trim_line_end(line);
          return LineReadStatus::kLine;
        }
        if (plain.eof()) {
          return LineReadStatus::kEof;
        }
        std::cerr << "Error reading input: " << path << "\n";
        return LineReadStatus::kError;
      }
      case Kind::kBzip2: {
        while (true) {
          const std::size_t newlinePos = bzipPending.find('\n');
          if (newlinePos != std::string::npos) {
            line.assign(bzipPending.data(), newlinePos);
            bzipPending.erase(0, newlinePos + 1);
            trim_line_end(line);
            return LineReadStatus::kLine;
          }

          if (bzipReachedEof) {
            if (bzipPending.empty()) {
              return LineReadStatus::kEof;
            }
            line.swap(bzipPending);
            trim_line_end(line);
            return LineReadStatus::kLine;
          }

          int bzError = BZ_OK;
          const int bytesRead = BZ2_bzRead(&bzError, bzip, bzipBuffer.data(),
                                           static_cast<int>(bzipBuffer.size()));
          if (bzError != BZ_OK && bzError != BZ_STREAM_END) {
            std::cerr << "Error reading bzip2 input: " << path << "\n";
            return LineReadStatus::kError;
          }
          if (bytesRead > 0) {
            bzipPending.append(bzipBuffer.data(), static_cast<std::size_t>(bytesRead));
          }
          if (bzError == BZ_STREAM_END) {
            bzipReachedEof = true;
          }
        }
      }
      case Kind::kNone:
        break;
    }
    return LineReadStatus::kError;
  }

  void close() {
    if (bzip) {
      int bzError = BZ_OK;
      BZ2_bzReadClose(&bzError, bzip);
      bzip = nullptr;
    }
    if (bzipFile) {
      std::fclose(bzipFile);
      bzipFile = nullptr;
    }
    if (plain.is_open()) {
      plain.close();
    }
    bzipReachedEof = false;
    bzipPending.clear();
    kind = Kind::kNone;
    path.clear();
  }

 private:
  static void trim_line_end(std::string &line) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
  }
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

void append_delimited_field(std::string &out, char delimiter,
                            std::string_view field) {
  const bool needsQuotes =
      delimiter == ',' &&
      field.find_first_of(",\"\r\n") != std::string_view::npos;
  if (!needsQuotes) {
    out.append(field.data(), field.size());
    return;
  }

  out.push_back('"');
  for (char c : field) {
    if (c == '"') {
      out.append("\"\"", 2);
    } else {
      out.push_back(c);
    }
  }
  out.push_back('"');
}

std::string serialize_delimited_row(const std::vector<std::string> &fields,
                                    char delimiter) {
  std::size_t reserveSize = fields.empty() ? 0 : fields.size() - 1;
  for (const auto &field : fields) {
    reserveSize += field.size();
  }

  std::string out;
  out.reserve(reserveSize);
  for (std::size_t i = 0; i < fields.size(); ++i) {
    if (i != 0) {
      out.push_back(delimiter);
    }
    append_delimited_field(out, delimiter, fields[i]);
  }
  return out;
}

bool find_column(const std::vector<std::string> &headerFields,
                 std::string_view targetName, std::size_t &columnIndex) {
  for (std::size_t i = 0; i < headerFields.size(); ++i) {
    if (to_lower_copy(trim_copy(headerFields[i])) == targetName) {
      columnIndex = i;
      return true;
    }
  }
  return false;
}

struct RawLine {
  std::string line;
};

struct ProcessedLine {
  std::string outLine;
  bool valid = false;
};

bool parse_header_and_delimiter(std::string_view headerLine, char &delimiter,
                                std::size_t &smilesColumn,
                                std::size_t &idColumn) {
  std::vector<std::string> fields;
  for (const char candidate : {',', '\t'}) {
    if (!parse_delimited_line(headerLine, candidate, fields)) {
      continue;
    }
    if (find_column(fields, "smiles", smilesColumn) &&
        find_column(fields, "id", idColumn)) {
      delimiter = candidate;
      return true;
    }
  }
  return false;
}

void worker_routine(const std::vector<RawLine> &inputs,
                    std::vector<ProcessedLine> &outputs, std::size_t start,
                    std::size_t end, std::size_t smilesColumn, std::size_t idColumn,
                    char delimiter) {
  std::vector<std::string> fields;
  for (std::size_t i = start; i < end; ++i) {
    auto &output = outputs[i];
    output.valid = false;
    output.outLine.clear();

    if (!parse_delimited_line(inputs[i].line, delimiter, fields)) {
      continue;
    }
    if (smilesColumn >= fields.size() || idColumn >= fields.size()) {
      continue;
    }

    const std::string &smiles = fields[smilesColumn];
    if (smiles.empty()) {
      continue;
    }

    RDKit::ROMol *rawMol = nullptr;
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
      output.outLine.clear();
      output.outLine.reserve(
          normalizedSmiles.size() + fields[idColumn].size() + 2);
      output.outLine.append(normalizedSmiles);
      output.outLine.push_back('\t');
      output.outLine.append(fields[idColumn]);
      output.outLine.push_back('\n');
      output.valid = true;
    } catch (...) {
    }
  }
}

bool process_chunk(const ChunkInput &chunk, const std::filesystem::path &outDir,
                   std::size_t innerThreads) {
  log_message("Removing stereochemistry: " + chunk.path);
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

  std::string headerLine;
  bool foundHeader = false;
  while (true) {
    const LineReadStatus status = reader.next_line(headerLine);
    if (status == LineReadStatus::kError) {
      reader.close();
      return false;
    }
    if (status == LineReadStatus::kEof) {
      std::cerr << "Chunk is empty: " << chunk.path << "\n";
      reader.close();
      return false;
    }
    if (trim_copy(headerLine).empty()) {
      continue;
    }
    strip_utf8_bom(headerLine);
    foundHeader = true;
    break;
  }
  if (!foundHeader) {
    std::cerr << "Failed to read header from chunk: " << chunk.path << "\n";
    reader.close();
    return false;
  }

  char delimiter = ',';
  std::size_t smilesColumn = 0;
  std::size_t idColumn = 0;
  if (!parse_header_and_delimiter(headerLine, delimiter, smilesColumn, idColumn)) {
    std::cerr << "Header does not contain both smiles and id columns: "
              << chunk.path << "\n";
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

  std::size_t readLines = 0;
  std::size_t processed = 0;
  std::size_t written = 0;
  std::size_t skipped = 0;
  bool ok = true;

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
    for (std::size_t i = 0; i < batchSize; ++i) {
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
      if (trim_copy(lineBuffer).empty()) {
        continue;
      }

      ++readLines;
      RawLine raw;
      raw.line = std::move(lineBuffer);
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

    if (innerThreads <= 1) {
      worker_routine(rawBatch, processedBatch, 0, rawBatch.size(),
                     smilesColumn, idColumn, delimiter);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(innerThreads);
      const std::size_t chunkSize =
          (rawBatch.size() + innerThreads - 1) / innerThreads;
      for (std::size_t t = 0; t < innerThreads; ++t) {
        const std::size_t start = t * chunkSize;
        const std::size_t end = std::min(start + chunkSize, rawBatch.size());
        if (start < end) {
          workers.emplace_back(worker_routine, std::cref(rawBatch),
                               std::ref(processedBatch), start, end,
                               smilesColumn, idColumn, delimiter);
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

void producerThread(const std::vector<ChunkInput> &inputs,
                    std::atomic<std::size_t> &nextIndex,
                    std::atomic<bool> &hadError,
                    const std::filesystem::path &outDir,
                    std::size_t innerThreads) {
  while (true) {
    const std::size_t idx = nextIndex.fetch_add(1);
    if (idx >= inputs.size()) {
      break;
    }
    if (!process_chunk(inputs[idx], outDir, innerThreads)) {
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
      producerThread(chunks, nextIndex, hadError, opts.outDir,
                     opts.innerThreads);
    });
  }
  for (auto &thread : producers) {
    thread.join();
  }

  const bool wroteTable = write_chunk_table(opts.outDir, chunks);
  return (hadError || !wroteTable) ? 1 : 0;
}
