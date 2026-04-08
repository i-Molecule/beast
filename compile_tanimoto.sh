#!/bin/bash
set -euo pipefail

CXX=${CXX:-g++}

$CXX -march=native -mtune=native -O3 -fopenmp -fPIC -shared -o src/chemical_screener/libtanimoto.so \
    src/chemical_screener/tanimoto_multi_query_overlap.cpp \
    src/chemical_screener/tanimoto_multi_query_overlap_packed.cpp \
    src/chemical_screener/tanimoto_single_query.cpp \
    src/chemical_screener/tanimoto_single_query_fp16.cpp \
    src/chemical_screener/tanimoto_single_query_packed_fp16.cpp \
    src/chemical_screener/tanimoto_single_query_filter.cpp \
    src/chemical_screener/tanimoto_single_query_filter_packed.cpp
