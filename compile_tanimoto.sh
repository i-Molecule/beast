#!/bin/bash
set -euo pipefail

CXX=${CXX:-g++}

$CXX -march=native -mtune=native -O3 -fopenmp -fPIC -shared -o src/beast/libtanimoto.so \
    src/beast/tanimoto_multi_query_overlap_packed.cpp \
    src/beast/tanimoto_multi_query_overlap_packed_with_scores.cpp \
    src/beast/tanimoto_single_query_packed_fp16.cpp \
    src/beast/tanimoto_single_query_filter_packed.cpp
