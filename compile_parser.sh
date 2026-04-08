#!/bin/bash
set -euo pipefail

CXX=${CXX:-g++}
CXXFLAGS="-O3 -std=c++20 -pthread"

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists htslib; then
  HTSLIB_FLAGS=$(pkg-config --cflags --libs htslib)
else
  HTSLIB_FLAGS="-lhts -lz"
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists zstd; then
  ZSTD_FLAGS=$(pkg-config --cflags --libs zstd)
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libzstd; then
  ZSTD_FLAGS=$(pkg-config --cflags --libs libzstd)
else
  ZSTD_FLAGS="-lzstd"
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists bzip2; then
  BZIP2_FLAGS=$(pkg-config --cflags --libs bzip2)
elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libbz2; then
  BZIP2_FLAGS=$(pkg-config --cflags --libs libbz2)
else
  BZIP2_FLAGS="-lbz2"
fi

if [ -z "${RDKIT_FLAGS:-}" ]; then
  if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists rdkit; then
    RDKIT_FLAGS=$(pkg-config --cflags --libs rdkit)
  elif command -v rdkit-config >/dev/null 2>&1; then
    RDKIT_FLAGS=$(rdkit-config --cflags --libs)
  elif [ -n "${CONDA_PREFIX:-}" ]; then
    RDKIT_FLAGS="-I${CONDA_PREFIX}/include/rdkit -L${CONDA_PREFIX}/lib \
      -lRDKitSmilesParse -lRDKitFingerprints -lRDKitDataStructs \
      -lRDKitGraphMol -lRDKitRDGeneral"
  else
    RDKIT_FLAGS=""
  fi
fi

$CXX $CXXFLAGS parser/deduplicate_zinc.cpp -o parser/deduplicate_zinc $HTSLIB_FLAGS
$CXX $CXXFLAGS parser/download_zinc.cpp -o parser/download_zinc -lz
$CXX $CXXFLAGS parser/remove_stereo.cpp -o parser/remove_stereo $HTSLIB_FLAGS $ZSTD_FLAGS $RDKIT_FLAGS
$CXX $CXXFLAGS parser/remove_stereo_enamine.cpp -o parser/remove_stereo_enamine $ZSTD_FLAGS $BZIP2_FLAGS $RDKIT_FLAGS
$CXX $CXXFLAGS parser/generate_fingerprints.cpp -o parser/generate_fingerprints $HTSLIB_FLAGS $ZSTD_FLAGS $RDKIT_FLAGS
