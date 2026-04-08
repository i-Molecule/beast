

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include <zstd.h>
#include <Geometry/point.h>
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/ForceFieldHelpers/UFF/UFF.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/SparseIntVect.h>


using namespace RDKit;

std::string normalize_absolute_path(const std::filesystem::path &path);

void log_message(const std::string &msg) {
  static std::mutex logMutex;
  std::lock_guard<std::mutex> lock(logMutex);
  std::cout << msg << std::endl;
}

enum class BitOrder { Big, Little };

// Saves a POD vector to disk as raw binary.
template <typename T>
void save_vector_as_binary(const std::vector<T> &data,
                           const std::string &filename) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Binary save requires trivially copyable elements");
  std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
}

// Appends a POD vector to disk as raw binary.
template <typename T>
void append_vector_as_binary(const std::vector<T> &data,
                             const std::string &filename) {
  static_assert(std::is_trivially_copyable_v<T>,
                "Binary save requires trivially copyable elements");
  std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::app);
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
}

// Packs an ExplicitBitVect into a reusable byte buffer with configurable bit order.
void packExplicitBitVect(const ExplicitBitVect &bv, BitOrder bitOrder,
                         std::vector<std::uint8_t> &out,
                         std::vector<int> &onBitsBuffer) {
  const unsigned int nbits = bv.getNumBits();
  const unsigned int nbytes = (nbits + 7) / 8;

  out.assign(nbytes, 0);
  if (nbytes == 0) {
    return;
  }

  onBitsBuffer.clear();
  bv.getOnBits(onBitsBuffer);

  const bool big = bitOrder == BitOrder::Big;
  for (const int idx : onBitsBuffer) {
    const unsigned int uidx = static_cast<unsigned int>(idx);
    const unsigned int byteIdx = uidx >> 3;
    const unsigned int bitInByte = uidx & 7U;
    const std::uint8_t mask =
        big ? static_cast<std::uint8_t>(1U << (7U - bitInByte))
            : static_cast<std::uint8_t>(1U << bitInByte);
    out[byteIdx] |= mask;
  }
}

constexpr std::uint32_t kRowBits = 19;
constexpr std::uint32_t kFrameBits = 13;
constexpr std::uint32_t kRowMask = (1u << kRowBits) - 1u;
constexpr std::uint32_t kMaxFrameId = (1u << kFrameBits) - 1u;

inline std::uint32_t pack_ref(std::uint32_t frameId, std::uint32_t rowInFrame) {
  return (frameId << kRowBits) | rowInFrame;
}

inline std::uint32_t ref_frame(std::uint32_t packed) {
  return packed >> kRowBits;
}

inline std::uint32_t ref_row(std::uint32_t packed) {
  return packed & kRowMask;
}

std::string build_npy_header(std::size_t rows, std::size_t cols) {
  std::ostringstream header;
  header << "{'descr': '|u1', 'fortran_order': False, 'shape': ("
         << rows << ", " << cols << "), }";
  std::string headerStr = header.str();
  headerStr.push_back('\n');

  const std::size_t preamble = 12;  // magic(6) + version(2) + header_len(4)
  const std::size_t padding = (16 - ((preamble + headerStr.size()) % 16)) % 16;
  headerStr.append(padding, ' ');
  return headerStr;
}

bool write_npy_header(std::ofstream &out, std::size_t rows, std::size_t cols) {
  const char magic[] = "\x93NUMPY";
  out.write(magic, sizeof(magic) - 1);
  const unsigned char version[] = {2, 0};
  out.write(reinterpret_cast<const char *>(version), sizeof(version));

  const std::string header = build_npy_header(rows, cols);
  const std::uint32_t headerLen = static_cast<std::uint32_t>(header.size());
  out.write(reinterpret_cast<const char *>(&headerLen), sizeof(headerLen));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  return static_cast<bool>(out);
}

bool write_npy_uint8_2d(const std::filesystem::path &path,
                        const std::vector<std::uint8_t> &data,
                        std::size_t rows, std::size_t cols) {
  if (rows * cols != data.size()) {
    return false;
  }
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  if (!write_npy_header(out, rows, cols)) {
    return false;
  }
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(out);
}

bool convert_bin_to_npy(const std::filesystem::path &binPath,
                        std::size_t fpBytes) {
  std::error_code ec;
  const std::uintmax_t binSize = std::filesystem::file_size(binPath, ec);
  if (ec) {
    std::cerr << "Failed to stat bin file: " << binPath << " (" << ec.message() << ")\n";
    return false;
  }
  if (fpBytes == 0 || binSize % fpBytes != 0) {
    std::cerr << "Invalid bin size for " << binPath << "\n";
    return false;
  }
  const std::size_t rows = static_cast<std::size_t>(binSize / fpBytes);
  std::vector<std::uint8_t> data(rows * fpBytes);
  std::ifstream in(binPath, std::ios::binary | std::ios::in);
  if (!in) {
    std::cerr << "Failed to open bin file: " << binPath << "\n";
    return false;
  }
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(data.size()));
  if (!in) {
    std::cerr << "Failed to read bin file: " << binPath << "\n";
    return false;
  }
  std::filesystem::path npyPath = binPath;
  npyPath.replace_extension(".npy");
  if (!write_npy_uint8_2d(npyPath, data, rows, fpBytes)) {
    std::cerr << "Failed to write npy file: " << npyPath << "\n";
    return false;
  }
  return true;
}

struct NpyRecord {
  std::uint32_t chunkId = 0;
  std::string npy;
  std::string refs;
  std::string zst;
  std::string frames;
  unsigned int fpSize = 0;
  unsigned int onBits = 0;
  std::size_t size = 0;
};

bool split_bin_refs_to_npy_parts(const std::filesystem::path &binPath,
                                 const std::filesystem::path &refsPath,
                                 std::size_t fpBytes,
                                 std::size_t maxRows,
                                 std::uint32_t chunkId,
                                 unsigned int fpSize,
                                 unsigned int onBits,
                                 const std::filesystem::path &zstPath,
                                 const std::filesystem::path &framesPath,
                                 std::vector<NpyRecord> &records,
                                 std::mutex &recordsMutex) {
  std::error_code ec;
  const std::uintmax_t binSize = std::filesystem::file_size(binPath, ec);
  if (ec) {
    std::cerr << "Failed to stat bin file: " << binPath << " (" << ec.message() << ")\n";
    return false;
  }
  if (fpBytes == 0 || binSize % fpBytes != 0) {
    std::cerr << "Invalid bin size for " << binPath << "\n";
    return false;
  }
  const std::size_t totalRows = static_cast<std::size_t>(binSize / fpBytes);
  const std::uintmax_t refsSize = std::filesystem::file_size(refsPath, ec);
  if (ec) {
    std::cerr << "Failed to stat refs file: " << refsPath << " (" << ec.message() << ")\n";
    return false;
  }
  if (refsSize != static_cast<std::uintmax_t>(totalRows * sizeof(std::uint32_t))) {
    std::cerr << "Refs size mismatch for " << refsPath << "\n";
    return false;
  }

  std::ifstream binIn(binPath, std::ios::binary | std::ios::in);
  std::ifstream refsIn(refsPath, std::ios::binary | std::ios::in);
  if (!binIn || !refsIn) {
    std::cerr << "Failed to open bin/refs for splitting: " << binPath << " | " << refsPath << "\n";
    return false;
  }

  const std::string base = binPath.stem().string();
  const std::filesystem::path binDir = binPath.parent_path();
  const std::filesystem::path refsDir = refsPath.parent_path();
  constexpr std::size_t kCopyBuffer = 8 * 1024 * 1024;
  std::vector<char> buffer(kCopyBuffer);

  std::size_t remaining = totalRows;
  std::size_t partIdx = 0;
  while (remaining > 0) {
    const std::size_t partRows = std::min(maxRows, remaining);
    const std::filesystem::path npyPath =
        binDir / (base + "_part_" + std::to_string(partIdx) + ".npy");
    const std::filesystem::path partRefsPath =
        refsDir / (refsPath.stem().string() + "_part_" + std::to_string(partIdx) + ".refs");

    std::ofstream npyOut(npyPath, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!npyOut || !write_npy_header(npyOut, partRows, fpBytes)) {
      std::cerr << "Failed to open/write npy part: " << npyPath << "\n";
      return false;
    }
    const std::size_t partBytes = partRows * fpBytes;
    std::size_t copied = 0;
    while (copied < partBytes) {
      const std::size_t toRead = std::min(kCopyBuffer, partBytes - copied);
      binIn.read(buffer.data(), static_cast<std::streamsize>(toRead));
      if (!binIn) {
        std::cerr << "Failed to read bin data for " << binPath << "\n";
        return false;
      }
      npyOut.write(buffer.data(), static_cast<std::streamsize>(toRead));
      if (!npyOut) {
        std::cerr << "Failed to write npy data for " << npyPath << "\n";
        return false;
      }
      copied += toRead;
    }

    std::ofstream refsOut(partRefsPath, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!refsOut) {
      std::cerr << "Failed to open refs part: " << partRefsPath << "\n";
      return false;
    }
    const std::size_t refsBytes = partRows * sizeof(std::uint32_t);
    std::size_t refsCopied = 0;
    while (refsCopied < refsBytes) {
      const std::size_t toRead = std::min(kCopyBuffer, refsBytes - refsCopied);
      refsIn.read(buffer.data(), static_cast<std::streamsize>(toRead));
      if (!refsIn) {
        std::cerr << "Failed to read refs data for " << refsPath << "\n";
        return false;
      }
      refsOut.write(buffer.data(), static_cast<std::streamsize>(toRead));
      if (!refsOut) {
        std::cerr << "Failed to write refs data for " << partRefsPath << "\n";
        return false;
      }
      refsCopied += toRead;
    }

    {
      std::lock_guard<std::mutex> lock(recordsMutex);
      records.push_back(NpyRecord{
          chunkId,
          npyPath.string(),
          partRefsPath.string(),
          zstPath.string(),
          framesPath.string(),
          fpSize,
          onBits,
          partRows,
      });
    }

    remaining -= partRows;
    ++partIdx;
  }

  return true;
}

std::string csv_escape(std::string_view value) {
  bool needsQuotes = false;
  for (char c : value) {
    if (c == '"' || c == ',' || c == '\n' || c == '\r') {
      needsQuotes = true;
      break;
    }
  }
  if (!needsQuotes) {
    return std::string(value);
  }
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (char c : value) {
    if (c == '"') {
      out.push_back('"');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

bool write_npy_chunks_csv(const std::filesystem::path &path,
                          std::vector<NpyRecord> records) {
  std::sort(records.begin(), records.end(),
            [](const NpyRecord &a, const NpyRecord &b) {
              if (a.fpSize != b.fpSize) {
                return a.fpSize < b.fpSize;
              }
              if (a.chunkId != b.chunkId) {
                return a.chunkId < b.chunkId;
              }
              if (a.onBits != b.onBits) {
                return a.onBits < b.onBits;
              }
              return a.npy < b.npy;
            });

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << "chunk_id,npy,refs,zst,frames,onbits,size\n";
  for (const auto &rec : records) {
    out << rec.chunkId << ","
        << csv_escape(normalize_absolute_path(rec.npy)) << ","
        << csv_escape(normalize_absolute_path(rec.refs)) << ","
        << csv_escape(normalize_absolute_path(rec.zst)) << ","
        << csv_escape(normalize_absolute_path(rec.frames)) << ","
        << rec.onBits << ","
        << rec.size << "\n";
  }
  return static_cast<bool>(out);
}

// Fixed-size SMILES structure - avoids heap allocations
// Max SMILES length: 162 chars, mean: 60 chars
struct FixedSmiles {
  static constexpr std::size_t kMaxLen = 162;
  static constexpr std::size_t kStorage = kMaxLen + 1;  // room for null terminator
  std::uint8_t len = 0;
  std::array<char, kStorage> data{};

  FixedSmiles() = default;

  explicit FixedSmiles(std::string_view sv) {
    if (!assign(sv)) {
      throw std::length_error("SMILES too long: " + std::to_string(sv.size()));
    }
  }

  bool assign(std::string_view sv) {
    if (sv.size() > kMaxLen) {
      return false;
    }
    len = static_cast<std::uint8_t>(sv.size());
    if (len > 0) {
      std::memcpy(data.data(), sv.data(), len);
    }
    if (len < data.size()) {
      std::memset(data.data() + len, 0, data.size() - len);
    }
    return true;
  }

  std::string_view view() const noexcept {
    return std::string_view(data.data(), len);
  }

  const char *c_str() const noexcept {
    return data.data();
  }

  bool operator==(const FixedSmiles &other) const noexcept {
    return len == other.len &&
           (len == 0 || std::memcmp(data.data(), other.data.data(), len) == 0);
  }

  bool operator==(std::string_view sv) const noexcept {
    return len == sv.size() &&
           (len == 0 || std::memcmp(data.data(), sv.data(), len) == 0);
  }
};

namespace std {
template <>
struct hash<FixedSmiles> {
  std::size_t operator()(const FixedSmiles &fs) const noexcept {
    // FNV-1a hash - 3-5x faster than std::hash for strings
    uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < fs.len; ++i) {
      h ^= static_cast<uint64_t>(static_cast<unsigned char>(fs.data[i]));
      h *= 1099511628211ULL;
    }
    return h;
  }
};
}  // namespace std

class ZstdLineReader {
 public:
  explicit ZstdLineReader(const std::string &path)
      : file_(path, std::ios::in | std::ios::binary),
        dstream_(ZSTD_createDStream()),
        inBuf_(ZSTD_DStreamInSize()),
        outBuf_(ZSTD_DStreamOutSize()) {
    if (!file_) {
      throw std::runtime_error("Failed to open ZSTD input file: " + path);
    }
    if (!dstream_) {
      throw std::runtime_error("Failed to create ZSTD decompression stream");
    }
    const std::size_t initResult = ZSTD_initDStream(dstream_);
    if (ZSTD_isError(initResult)) {
      throw std::runtime_error(std::string("ZSTD init error: ") +
                               ZSTD_getErrorName(initResult));
    }
  }

  ~ZstdLineReader() {
    if (dstream_) {
      ZSTD_freeDStream(dstream_);
    }
  }

  bool nextLine(std::string_view &line, std::uint32_t &packedRef) {
    if (start_ > kTrimThreshold) {
      buffer_.erase(0, start_);
      start_ = 0;
    }

    while (true) {
      if (start_ < buffer_.size()) {
        const std::size_t newlinePos = buffer_.find('\n', start_);
        if (newlinePos != std::string::npos) {
          line = std::string_view(buffer_.data() + start_,
                                  newlinePos - start_);
          if (rowInFrame_ > kRowMask) {
            throw std::runtime_error("ZSTD frame has too many rows");
          }
          packedRef = pack_ref(frameId_, rowInFrame_++);
          start_ = newlinePos + 1;
          if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
          }
          return true;
        }
      }

      if (eof_) {
        if (start_ < buffer_.size()) {
          line = std::string_view(buffer_.data() + start_,
                                  buffer_.size() - start_);
          if (rowInFrame_ > kRowMask) {
            throw std::runtime_error("ZSTD frame has too many rows");
          }
          packedRef = pack_ref(frameId_, rowInFrame_++);
          start_ = buffer_.size();
          if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
          }
          return true;
        }
        return false;
      }

      if (pendingNewFrame_ && start_ >= buffer_.size()) {
        const std::uint64_t pendingOffset = nextFrameOffset_;
        pendingNewFrame_ = false;
        buffer_.clear();
        start_ = 0;
        if (!decompressMore()) {
          eof_ = true;
          return false;
        }
        ++frameId_;
        if (frameId_ > kMaxFrameId) {
          throw std::runtime_error("ZSTD has too many frames for 11-bit id");
        }
        rowInFrame_ = 0;
        frameOffsets_.push_back(pendingOffset);
        continue;
      }

      if (!decompressMore()) {
        eof_ = true;
      }
    }
  }

  const std::vector<std::uint64_t> &frameOffsets() const {
    return frameOffsets_;
  }

 private:
  bool decompressMore() {
    while (true) {
      if (input_.pos == input_.size) {
        file_.read(inBuf_.data(), static_cast<std::streamsize>(inBuf_.size()));
        const std::streamsize got = file_.gcount();
        input_.src = inBuf_.data();
        input_.size = static_cast<std::size_t>(got);
        input_.pos = 0;
      }

      ZSTD_outBuffer output{outBuf_.data(), outBuf_.size(), 0};
      const std::size_t prevPos = input_.pos;
      const std::size_t ret = ZSTD_decompressStream(dstream_, &output, &input_);
      compressedConsumed_ += input_.pos - prevPos;
      if (ZSTD_isError(ret)) {
        throw std::runtime_error(std::string("ZSTD decompression error: ") +
                                 ZSTD_getErrorName(ret));
      }
      if (ret == 0) {
        pendingNewFrame_ = true;
        nextFrameOffset_ = compressedConsumed_;
      }

      if (output.pos > 0) {
        buffer_.append(outBuf_.data(), output.pos);
        return true;
      }

      if (input_.size == 0 && input_.pos == 0) {
        return false;
      }
      if (ret == 0) {
        return true;
      }
    }
  }

  static constexpr std::size_t kTrimThreshold = 1 << 20;

  std::ifstream file_;
  ZSTD_DStream *dstream_ = nullptr;
  std::vector<char> inBuf_;
  std::vector<char> outBuf_;
  ZSTD_inBuffer input_{nullptr, 0, 0};
  std::string buffer_;
  std::size_t start_ = 0;
  std::uint32_t frameId_ = 0;
  std::uint32_t rowInFrame_ = 0;
  std::vector<std::uint64_t> frameOffsets_{0};
  std::uint64_t compressedConsumed_ = 0;
  bool eof_ = false;
  bool pendingNewFrame_ = false;
  std::uint64_t nextFrameOffset_ = 0;
};


struct OutputPaths {
  std::filesystem::path fpDir;
  std::filesystem::path idxDir;
};

struct ChunkInput {
  std::string path;
  std::string baseName;
  std::uint32_t chunkId = 0;
};

// constexpr std::array<unsigned int, 3> kDefaultFingerprintSizes = {64, 128, 256};
constexpr std::array<unsigned int, 4> kDefaultFingerprintSizes = {64, 128, 256, 512};
constexpr unsigned int kFingerprintRadius = 2;
constexpr std::size_t kMaxFpInChunk = 1'000'000;
constexpr std::size_t kDefaultBufferBytes = 32'000'000;

struct FingerprintSpec {
  unsigned int fpSize = 0;
  unsigned int fpBytes = 0;
  std::size_t bucketCount = 0;
  std::string fpDirName;
  std::string idxDirName;
  std::filesystem::path fpOutputDir;
  std::filesystem::path idxOutputDir;
};

std::vector<FingerprintSpec> buildFingerprintSpecs(
    const std::vector<unsigned int> &requestedSizes = {}) {
  const std::size_t specCount =
      requestedSizes.empty() ? kDefaultFingerprintSizes.size() : requestedSizes.size();
  std::vector<FingerprintSpec> specs;
  specs.reserve(specCount);
  auto append_spec = [&specs](unsigned int size) {
    FingerprintSpec spec;
    spec.fpSize = size;
    spec.fpBytes = (size + 7) / 8;
    spec.bucketCount = static_cast<std::size_t>(size) + 1;
    spec.fpDirName = "fp_" + std::to_string(size);
    spec.idxDirName = "idx_" + std::to_string(size);
    specs.emplace_back(std::move(spec));
  };

  if (requestedSizes.empty()) {
    for (const unsigned int size : kDefaultFingerprintSizes) {
      append_spec(size);
    }
  } else {
    for (const unsigned int size : requestedSizes) {
      append_spec(size);
    }
  }
  return specs;
}

void save_offsets(const std::vector<std::uint32_t> &offsets,
                  const std::filesystem::path &filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Failed to open reference file for writing: " + filename.string());
  }
  out.write(reinterpret_cast<const char *>(offsets.data()),
            static_cast<std::streamsize>(offsets.size() * sizeof(std::uint32_t)));
}

void append_offsets(const std::vector<std::uint32_t> &offsets,
                    const std::filesystem::path &filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::app);
  if (!out) {
    throw std::runtime_error("Failed to open reference file for appending: " + filename.string());
  }
  out.write(reinterpret_cast<const char *>(offsets.data()),
            static_cast<std::streamsize>(offsets.size() * sizeof(std::uint32_t)));
}

void save_frame_index(const std::vector<std::uint64_t> &offsets,
                      const std::filesystem::path &filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Failed to open frame index file for writing: " + filename.string());
  }
  out.write(reinterpret_cast<const char *>(offsets.data()),
            static_cast<std::streamsize>(offsets.size() * sizeof(std::uint64_t)));
}

struct ProgramOptions {
  std::size_t workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::string> inputs;
  std::string chunkTable;
  std::string fpDir = ".";
  std::string idxDir = ".";
  std::string tableDir;
  std::size_t bufferBytes = kDefaultBufferBytes;
  std::string unfinishedChunkIdsPath;
  std::vector<unsigned int> fingerprintSizes;
};

bool parse_positive_size(const std::string &value, std::size_t &out) {
  try {
    std::size_t idx = 0;
    const unsigned long long parsed = std::stoull(value, &idx, 10);
    if (idx != value.size() || parsed == 0) {
      return false;
    }
    out = static_cast<std::size_t>(parsed);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool is_valid_fingerprint_size(unsigned int size) {
  return size > 0 && (size % 8) == 0;
}

std::string default_fingerprint_sizes_string() {
  std::ostringstream out;
  for (std::size_t i = 0; i < kDefaultFingerprintSizes.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << kDefaultFingerprintSizes[i];
  }
  return out.str();
}

bool parse_fingerprint_sizes_arg(const std::string &value,
                                 std::vector<unsigned int> &out) {
  std::unordered_set<unsigned int> requested;
  std::stringstream ss(value);
  std::string token;
  bool sawValue = false;

  while (std::getline(ss, token, ',')) {
    const auto first = std::find_if_not(token.begin(), token.end(),
                                        [](unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(token.rbegin(), token.rend(),
                                       [](unsigned char c) { return std::isspace(c); })
                          .base();
    if (first >= last) {
      return false;
    }

    std::size_t parsed = 0;
    if (!parse_positive_size(std::string(first, last), parsed) ||
        parsed > std::numeric_limits<unsigned int>::max()) {
      return false;
    }

    const unsigned int size = static_cast<unsigned int>(parsed);
    if (!is_valid_fingerprint_size(size)) {
      return false;
    }
    requested.insert(size);
    sawValue = true;
  }

  if (!sawValue || requested.empty()) {
    return false;
  }

  out.clear();
  out.reserve(requested.size());
  for (const unsigned int size : requested) {
    out.push_back(size);
  }
  std::sort(out.begin(), out.end());
  return true;
}

void print_usage(const char *progName) {
  std::cerr << "Usage: " << progName
            << " --chunk-table <csv> [-w|--workers <num>] "
               "[--fp-sizes <csv>] "
               "[--fp-dir <dir>] [--idx-dir <dir>] [--table-dir <dir>] "
               "[--buffer-bytes <num>] "
               "[--unfinished-chunks <txt>]\n";
}

bool parse_arguments(int argc, char *argv[], ProgramOptions &opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--workers" || arg == "-w") {
      if (i + 1 >= argc || !parse_positive_size(argv[i + 1], opts.workers)) {
        std::cerr << "Invalid worker count\n";
        return false;
      }
      ++i;
    } else if (arg == "--fp-sizes") {
      if (i + 1 >= argc) {
        std::cerr << "--fp-sizes requires a comma-separated list\n";
        return false;
      }
      if (!parse_fingerprint_sizes_arg(argv[++i], opts.fingerprintSizes)) {
        std::cerr << "Invalid --fp-sizes value. Expected a comma-separated list "
                     "of positive multiples of 8. Default sizes: "
                  << default_fingerprint_sizes_string() << "\n";
        return false;
      }
    } else if (arg == "--fp-dir") {
      if (i + 1 >= argc) {
        std::cerr << "--fp-dir requires a path\n";
        return false;
      }
      opts.fpDir = argv[++i];
    } else if (arg == "--buffer-bytes") {
      if (i + 1 >= argc || !parse_positive_size(argv[i + 1], opts.bufferBytes)) {
        std::cerr << "Invalid buffer size\n";
        return false;
      }
      ++i;
    } else if (arg == "--idx-dir") {
      if (i + 1 >= argc) {
        std::cerr << "--idx-dir requires a path\n";
        return false;
      }
      opts.idxDir = argv[++i];
    } else if (arg == "--table-dir") {
      if (i + 1 >= argc) {
        std::cerr << "--table-dir requires a path\n";
        return false;
      }
      opts.tableDir = argv[++i];
    } else if (arg == "--chunk-table") {
      if (i + 1 >= argc) {
        std::cerr << "--chunk-table requires a path\n";
        return false;
      }
      opts.chunkTable = argv[++i];
    } else if (arg == "--unfinished-chunks" || arg == "--unfinished-chunk-ids") {
      if (i + 1 >= argc) {
        std::cerr << "--unfinished-chunks requires a path\n";
        return false;
      }
      opts.unfinishedChunkIdsPath = argv[++i];
    } else if (!arg.empty() && arg.front() == '-') {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    } else {
      opts.inputs.emplace_back(arg);
    }
  }
  if (!opts.inputs.empty()) {
    std::cerr << "Positional inputs are not supported; use --chunk-table.\n";
    return false;
  }
  if (opts.chunkTable.empty()) {
    std::cerr << "--chunk-table is required.\n";
    return false;
  }
  return true;
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
    } catch (const std::exception &) {
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
    if (!relPathStr.empty()) {
      chunk.baseName = std::filesystem::path(relPathStr).stem().string();
    }
    if (chunk.baseName.empty()) {
      chunk.baseName = absPath.stem().string();
    }
    out.emplace_back(std::move(chunk));
  }

  if (out.empty()) {
    std::cerr << "Chunk table did not contain any entries: " << tablePath << "\n";
    return false;
  }
  return true;
}

bool read_chunk_ids_file(const std::string &path,
                         std::unordered_set<std::uint32_t> &out) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "Failed to open unfinished chunk id file: " << path << "\n";
    return false;
  }
  out.clear();
  std::string line;
  while (std::getline(input, line)) {
    std::string trimmed = trim_copy(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    unsigned long long parsedId = 0;
    try {
      parsedId = std::stoull(trimmed);
    } catch (const std::exception &) {
      std::cerr << "Invalid chunk id in unfinished list: " << trimmed << "\n";
      return false;
    }
    if (parsedId > std::numeric_limits<std::uint32_t>::max()) {
      std::cerr << "Chunk id too large in unfinished list: " << trimmed << "\n";
      return false;
    }
    out.insert(static_cast<std::uint32_t>(parsedId));
  }
  if (out.empty()) {
    std::cerr << "Unfinished chunk id file was empty: " << path << "\n";
    return false;
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

bool configure_fingerprint_directories(const OutputPaths &outPaths,
                                       std::vector<FingerprintSpec> &specs) {
  for (auto &spec : specs) {
    spec.fpOutputDir = outPaths.fpDir / spec.fpDirName;
    spec.idxOutputDir = outPaths.idxDir / spec.idxDirName;

    const std::string fpDesc = "fp subdir " + spec.fpDirName;
    if (!ensure_directory(spec.fpOutputDir, fpDesc.c_str())) {
      return false;
    }
    const std::string idxDesc = "idx subdir " + spec.idxDirName;
    if (!ensure_directory(spec.idxOutputDir, idxDesc.c_str())) {
      return false;
    }
  }
  return true;
}

std::string make_chunk_onbits_base(std::uint32_t chunkId, unsigned int onBits) {
  std::ostringstream base;
  base << "chunk_" << chunkId << "_onbits_" << onBits;
  return base.str();
}

void remove_output_file(const std::filesystem::path &path,
                        std::size_t &removed,
                        std::size_t &errors) {
  std::error_code ec;
  const bool didRemove = std::filesystem::remove(path, ec);
  if (ec) {
    ++errors;
    std::cerr << "Failed to remove " << path << " (" << ec.message() << ")\n";
    return;
  }
  if (didRemove) {
    ++removed;
  }
}

void remove_outputs_for_chunks(
    const std::unordered_set<std::uint32_t> &chunkIds,
    const std::vector<FingerprintSpec> &specs,
    const std::filesystem::path &frameIndexDir) {
  std::size_t removed = 0;
  std::size_t errors = 0;
  for (const std::uint32_t chunkId : chunkIds) {
    const std::filesystem::path framesPath =
        frameIndexDir / ("chunk_" + std::to_string(chunkId) + ".frames");
    remove_output_file(framesPath, removed, errors);
    for (const auto &spec : specs) {
      for (unsigned int bitCount = 0; bitCount <= spec.fpSize; ++bitCount) {
        const std::string base = make_chunk_onbits_base(chunkId, bitCount);
        remove_output_file(spec.fpOutputDir / (base + ".bin"), removed, errors);
        remove_output_file(spec.fpOutputDir / (base + ".npy"), removed, errors);
        remove_output_file(spec.idxOutputDir / (base + ".refs"), removed, errors);
      }
    }
  }
  log_message("Removed " + std::to_string(removed) +
              " output files for unfinished chunks" +
              (errors ? (" (" + std::to_string(errors) + " errors)") : ""));
}

struct WriterAccumulator {
  std::vector<std::uint8_t> bin;
  std::vector<std::uint32_t> offsets;
  std::size_t count = 0;
  bool wrote = false;
};

std::filesystem::path make_bin_path(const FingerprintSpec &spec,
                                    const ChunkInput &chunk,
                                    unsigned int onBits) {
  std::ostringstream base;
  base << "chunk_" << chunk.chunkId
       << "_onbits_" << onBits;
  return spec.fpOutputDir / (base.str() + ".bin");
}

std::filesystem::path make_refs_path(const FingerprintSpec &spec,
                                     const ChunkInput &chunk,
                                     unsigned int onBits) {
  std::ostringstream base;
  base << "chunk_" << chunk.chunkId
       << "_onbits_" << onBits;
  return spec.idxOutputDir / (base.str() + ".refs");
}

void flushAccumulator(const FingerprintSpec &spec, const ChunkInput &chunk,
                      unsigned int onBits, WriterAccumulator &acc) {
  if (acc.count == 0) {
    return;
  }
  const std::filesystem::path binName = make_bin_path(spec, chunk, onBits);
  const std::filesystem::path refName = make_refs_path(spec, chunk, onBits);

  if (!acc.wrote) {
    save_vector_as_binary(acc.bin, binName.string());
    save_offsets(acc.offsets, refName);
  } else {
    append_vector_as_binary(acc.bin, binName.string());
    append_offsets(acc.offsets, refName);
  }
  // log_message("Appended chunk for fp" + std::to_string(spec.fpSize) + ": " +
              // binName.string() + " | " + refName.string());
  acc.wrote = true;

  acc.count = 0;
  acc.bin.clear();
  acc.offsets.clear();
}

bool process_input_chunk(const ChunkInput &chunk,
                         const std::vector<FingerprintSpec> &specs,
                         std::size_t bufferBytes,
                         const std::filesystem::path &frameIndexDir,
                         std::vector<NpyRecord> &records,
                         std::mutex &recordsMutex) {
  log_message("Worker reading ZSTD chunk: " + chunk.path);

  unsigned int maxFpSize = 0;
  unsigned int maxFpBytes = 0;
  for (const auto &spec : specs) {
    maxFpSize = std::max(maxFpSize, spec.fpSize);
    maxFpBytes = std::max(maxFpBytes, spec.fpBytes);
  }

  std::vector<std::uint8_t> packedFp;
  packedFp.reserve(maxFpBytes);
  std::vector<int> onBitsBuffer;
  onBitsBuffer.reserve(maxFpSize);
  std::vector<std::vector<WriterAccumulator>> buckets(specs.size());
  for (std::size_t specIdx = 0; specIdx < specs.size(); ++specIdx) {
    buckets[specIdx].resize(specs[specIdx].bucketCount);
  }

  std::size_t processed = 0;
  try {
    ZstdLineReader reader(chunk.path);
    std::string_view line;
    std::uint32_t ref = 0;
    while (reader.nextLine(line, ref)) {
      if (line.empty()) {
        continue;
      }
      const std::size_t tabPos = line.find('\t');
      const std::string_view smilesView =
          (tabPos == std::string_view::npos) ? line : line.substr(0, tabPos);
      if (smilesView.empty() || smilesView.size() > FixedSmiles::kMaxLen) {
        continue;
      }
      FixedSmiles smiles;
      if (!smiles.assign(smilesView)) {
        continue;
      }

      RDKit::ROMol *rawMol = nullptr;
      try {
        rawMol = RDKit::SmilesToMol(smiles.c_str()); 
      } catch (const RDKit::MolSanitizeException &e) {
        log_message("RDKit MolSanitizeException for chunk " + std::to_string(chunk.chunkId) +
                    " frame " + std::to_string(ref_frame(ref)) +
                    " row " + std::to_string(ref_row(ref)) + ": " + e.what());
        continue;
      } catch (const std::exception &e) {
        log_message("RDKit exception for chunk " + std::to_string(chunk.chunkId) +
                    " frame " + std::to_string(ref_frame(ref)) +
                    " row " + std::to_string(ref_row(ref)) + ": " + e.what());
        continue;
      } catch (...) {
        log_message("Unknown RDKit exception for chunk " + std::to_string(chunk.chunkId) +
                    " frame " + std::to_string(ref_frame(ref)) +
                    " row " + std::to_string(ref_row(ref)));
        continue;
      }

      std::unique_ptr<RDKit::ROMol> m{rawMol};
      if (!m) {
        continue;
      }

      for (std::size_t specIdx = 0; specIdx < specs.size(); ++specIdx) {
        const auto &spec = specs[specIdx];
        const std::unique_ptr<ExplicitBitVect> fp(
            RDKit::MorganFingerprints::getFingerprintAsBitVect(
                *m, kFingerprintRadius, spec.fpSize));
        if (!fp) {
          continue;
        }
        const unsigned int onBits = fp->getNumOnBits();
        if (onBits > spec.fpSize) {
          continue;
        }

        packExplicitBitVect(*fp, BitOrder::Big, packedFp, onBitsBuffer);
        auto &bucket = buckets[specIdx][onBits];
        if (bucket.bin.empty()) {
          const std::size_t entryBytes =
              static_cast<std::size_t>(spec.fpBytes) + sizeof(std::uint32_t);
          const std::size_t reserveEntries =
              std::max<std::size_t>(1, bufferBytes / entryBytes);
          bucket.bin.reserve(reserveEntries * static_cast<std::size_t>(spec.fpBytes));
          bucket.offsets.reserve(reserveEntries);
        }
        bucket.bin.insert(bucket.bin.end(), packedFp.begin(),
                          packedFp.begin() + spec.fpBytes);
        bucket.offsets.emplace_back(ref);
        ++bucket.count;

        const std::size_t bufferedBytes =
            bucket.bin.size() + bucket.offsets.size() * sizeof(std::uint32_t);
        if (bufferedBytes >= bufferBytes) {
          flushAccumulator(spec, chunk, onBits, bucket);
        }
      }
      ++processed;
    }
    const std::filesystem::path frameIndexName =
        frameIndexDir / ("chunk_" + std::to_string(chunk.chunkId) + ".frames");
    save_frame_index(reader.frameOffsets(), frameIndexName);
    log_message("Saved frame index: " + frameIndexName.string());
  } catch (const std::exception &e) {
    std::cerr << "Failed to read ZSTD chunk " << chunk.path << ": " << e.what() << "\n";
    return false;
  }

  for (std::size_t specIdx = 0; specIdx < specs.size(); ++specIdx) {
    for (unsigned int bitCount = 0; bitCount <= specs[specIdx].fpSize; ++bitCount) {
      flushAccumulator(specs[specIdx], chunk, bitCount, buckets[specIdx][bitCount]);
    }
  }

  for (std::size_t specIdx = 0; specIdx < specs.size(); ++specIdx) {
    const auto &spec = specs[specIdx];
    for (unsigned int bitCount = 0; bitCount <= spec.fpSize; ++bitCount) {
      const auto &bucket = buckets[specIdx][bitCount];
      if (!bucket.wrote) {
        continue;
      }
      const std::filesystem::path binName = make_bin_path(spec, chunk, bitCount);
      const std::filesystem::path refsName = make_refs_path(spec, chunk, bitCount);
      const std::filesystem::path framesPath =
          frameIndexDir / ("chunk_" + std::to_string(chunk.chunkId) + ".frames");
      std::error_code sizeEc;
      const std::uintmax_t binSize = std::filesystem::file_size(binName, sizeEc);
      if (sizeEc || spec.fpBytes == 0 || binSize % spec.fpBytes != 0) {
        std::cerr << "Failed to stat bin file: " << binName << "\n";
        continue;
      }
      const std::size_t rows = static_cast<std::size_t>(binSize / spec.fpBytes);
      if (rows > kMaxFpInChunk) {
        if (!split_bin_refs_to_npy_parts(binName,
                                         refsName,
                                         spec.fpBytes,
                                         kMaxFpInChunk,
                                         chunk.chunkId,
                                         spec.fpSize,
                                         bitCount,
                                         chunk.path,
                                         framesPath,
                                         records,
                                         recordsMutex)) {
          std::cerr << "Failed to split bin/refs: " << binName << "\n";
          continue;
        }
        std::error_code rmec;
        std::filesystem::remove(binName, rmec);
        std::filesystem::remove(refsName, rmec);
        if (rmec) {
          std::cerr << "Failed to remove bin/refs file: " << binName
                    << " (" << rmec.message() << ")\n";
        }
      } else {
        if (!convert_bin_to_npy(binName, spec.fpBytes)) {
          std::cerr << "Failed to convert bin to npy: " << binName << "\n";
          continue;
        }
        {
          std::filesystem::path npyPath = binName;
          npyPath.replace_extension(".npy");
          std::lock_guard<std::mutex> lock(recordsMutex);
          records.push_back(NpyRecord{
              chunk.chunkId,
              npyPath.string(),
              refsName.string(),
              chunk.path,
              framesPath.string(),
              spec.fpSize,
              bitCount,
              rows,
          });
        }
        std::error_code rmec;
        std::filesystem::remove(binName, rmec);
        if (rmec) {
          std::cerr << "Failed to remove bin file: " << binName
                    << " (" << rmec.message() << ")\n";
        }
      }
    }
  }

  log_message("Worker finished chunk " + chunk.path + " (" +
              std::to_string(processed) + " rows)");
  return true;
}

void workerThread(const std::vector<ChunkInput> &inputs,
                  std::atomic<std::size_t> &nextIndex,
                  const std::vector<FingerprintSpec> &fingerprintSpecs,
                  std::size_t bufferBytes,
                  const std::filesystem::path &frameIndexDir,
                  std::vector<NpyRecord> &records,
                  std::mutex &recordsMutex) {
  while (true) {
    const std::size_t idx = nextIndex.fetch_add(1);
    if (idx >= inputs.size()) {
      break;
    }
    process_input_chunk(inputs[idx],
                        fingerprintSpecs,
                        bufferBytes,
                        frameIndexDir,
                        records,
                        recordsMutex);
  }
}

void gen_fp(const std::vector<ChunkInput> &inputs,
            std::size_t workerCount,
            const std::vector<FingerprintSpec> &fingerprintSpecs,
            const std::filesystem::path &frameIndexDir,
            std::size_t bufferBytes,
            std::vector<NpyRecord> &records,
            std::mutex &recordsMutex) {
  const std::size_t actualWorkers = std::max<std::size_t>(1, workerCount);
  std::atomic<std::size_t> nextIndex{0};
  std::vector<std::thread> workers;
  workers.reserve(actualWorkers);
  for (std::size_t i = 0; i < actualWorkers; ++i) {
    workers.emplace_back([&inputs,
                          &nextIndex,
                          &fingerprintSpecs,
                          &frameIndexDir,
                          bufferBytes,
                          &records,
                          &recordsMutex]() {
      workerThread(inputs,
                   nextIndex,
                   fingerprintSpecs,
                   bufferBytes,
                   frameIndexDir,
                   records,
                   recordsMutex);
    });
  }
  for (auto &t : workers) {
    t.join();
  }
}

int main(int argc, char *argv[]) {
  ProgramOptions opts;
  if (!parse_arguments(argc, argv, opts)) {
    print_usage(argv[0]);
    return 1;
  }

  std::vector<ChunkInput> inputFiles;
  if (!read_chunk_table(opts.chunkTable, inputFiles)) {
    return 1;
  }

  const auto wall_start = std::chrono::system_clock::now();
  const auto start_time = std::chrono::steady_clock::now();

  const std::time_t start_time_t = std::chrono::system_clock::to_time_t(wall_start);
  std::cout << "Start time: "
            << std::put_time(std::localtime(&start_time_t), "%Y-%m-%d %H:%M:%S")
            << std::endl;

  OutputPaths outPaths{opts.fpDir, opts.idxDir};
  if (!ensure_directory(outPaths.fpDir, "fp-dir") ||
      !ensure_directory(outPaths.idxDir, "idx-dir")) {
    return 1;
  }
  const std::filesystem::path tableDir =
      opts.tableDir.empty() ? outPaths.idxDir : std::filesystem::path(opts.tableDir);
  if (!ensure_directory(tableDir, "table-dir")) {
    return 1;
  }

  auto fingerprintSpecs = buildFingerprintSpecs(opts.fingerprintSizes);
  if (!configure_fingerprint_directories(outPaths, fingerprintSpecs)) {
    return 1;
  }

  if (!opts.unfinishedChunkIdsPath.empty()) {
    std::unordered_set<std::uint32_t> unfinishedChunkIds;
    if (!read_chunk_ids_file(opts.unfinishedChunkIdsPath, unfinishedChunkIds)) {
      return 1;
    }
    remove_outputs_for_chunks(unfinishedChunkIds, fingerprintSpecs, outPaths.idxDir);

    std::unordered_set<std::uint32_t> tableIds;
    tableIds.reserve(inputFiles.size());
    for (const auto &chunk : inputFiles) {
      tableIds.insert(chunk.chunkId);
    }
    std::size_t missing = 0;
    for (const auto id : unfinishedChunkIds) {
      if (tableIds.find(id) == tableIds.end()) {
        ++missing;
      }
    }
    if (missing > 0) {
      std::cerr << "Warning: " << missing
                << " unfinished chunk ids were not present in the chunk table.\n";
    }

    std::vector<ChunkInput> filtered;
    filtered.reserve(inputFiles.size());
    for (const auto &chunk : inputFiles) {
      if (unfinishedChunkIds.find(chunk.chunkId) != unfinishedChunkIds.end()) {
        filtered.push_back(chunk);
      }
    }
    if (filtered.empty()) {
      std::cerr << "No chunk ids from " << opts.unfinishedChunkIdsPath
                << " matched the chunk table.\n";
      return 1;
    }
    log_message("Processing " + std::to_string(filtered.size()) +
                " unfinished chunks from " + opts.unfinishedChunkIdsPath);
    inputFiles = std::move(filtered);
  }

  std::vector<NpyRecord> npyRecords;
  npyRecords.reserve(inputFiles.size() * 8);
  std::mutex recordsMutex;

  gen_fp(inputFiles,
         opts.workers,
         fingerprintSpecs,
         outPaths.idxDir,
         opts.bufferBytes,
         npyRecords,
         recordsMutex);

  std::unordered_map<unsigned int, std::vector<NpyRecord>> recordsBySize;
  recordsBySize.reserve(fingerprintSpecs.size());
  for (auto &rec : npyRecords) {
    recordsBySize[rec.fpSize].push_back(std::move(rec));
  }

  for (const auto &spec : fingerprintSpecs) {
    const std::filesystem::path tablePath =
        tableDir / ("table_" + std::to_string(spec.fpSize) + ".csv");
    auto it = recordsBySize.find(spec.fpSize);
    const std::vector<NpyRecord> empty;
    const auto &records = (it == recordsBySize.end()) ? empty : it->second;
    if (!write_npy_chunks_csv(tablePath, records)) {
      std::cerr << "Failed to write npy chunks table: " << tablePath << "\n";
      return 1;
    }
  }

  const auto wall_end = std::chrono::system_clock::now();
  const auto end_time = std::chrono::steady_clock::now();
  const std::time_t end_time_t = std::chrono::system_clock::to_time_t(wall_end);

  std::cout << "Finish time: "
            << std::put_time(std::localtime(&end_time_t), "%Y-%m-%d %H:%M:%S")
            << std::endl;

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
          .count();
  std::cout << "Elapsed: " << elapsed_ms / 1000.0 << " seconds" << std::endl;
}
