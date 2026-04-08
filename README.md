# chemical-screener

`chemical-screener` builds searchable fingerprint tables from large SMILES libraries
such as ZINC or Enamine REAL and queries them from Python.

## Requirements

- Python `3.10` to `3.12`.
- [Pixi](https://pixi.sh).

## Setup

Install Pixi, create the environment, and build the native components:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
pixi run build
```

`pixi run build` compiles:

- the parser binaries in `parser/`
- `src/chemical_screener/libtanimoto.so`

Use `pixi shell` for an interactive environment, or prefix commands with
`pixi run ...`.

## Quickstart

The repository already includes a small generated dataset under `parser/data/`.
That is the fastest way to confirm the Python API works.

Optional sanity check:

```bash
pixi run python -m pytest test/tests.py -q
```

Minimal Python example:

```python
import numpy as np

from chemical_screener.database import ZINC
from chemical_screener.run import get_sim_compounds, get_sim_scores

db = ZINC("parser/data/tables/table_128.csv", max_files=3)

# The screening API expects an unpacked 0/1 bit vector.
# For a quick smoke test we reuse a fingerprint that is already in the database.
packed_query = np.load(db.fp_files[0], mmap_mode="r")[0]
query_bits = np.unpackbits(packed_query, bitorder="big").astype(np.uint8)

scores = get_sim_scores(
    x_b=db,
    x_q=query_bits,
    num_workers=1,
    threads_per_worker=1,
)

hits = get_sim_compounds(
    x_b=db,
    x_q=query_bits,
    lower_bound=0.8,
    upper_bound=1.0,
    return_smiles=True,
    num_workers=1,
    threads_per_worker=1,
)

print(scores.shape)
print(next(iter(hits.items())))
```

Notes:

- `ZINC(...)` expects a generated database manifest such as `table_128.csv`, not
  the raw `chunk_table.csv` from the download step.
- `get_sim_scores` currently supports only one query fingerprint.
- `get_sim_compounds` accepts either a single query or a stack of queries.

## Output Files

The pipeline produces two different CSV formats:

1. `chunk_table.csv`

Used by the C++ parser tools as input between preprocessing stages.

```csv
chunk_id,abs_path,rel_path
0,/absolute/path/to/chunk_0.gz,chunk_0.gz
1,/absolute/path/to/chunk_1.gz,chunk_1.gz
```

2. `table_<fp_size>.csv`

Produced by `generate_fingerprints` and used by the Python API.

```csv
chunk_id,npy,refs,zst,frames,onbits,size
0,fp/fp_128/chunk_0_onbits_11.npy,idx/idx_128/chunk_0_onbits_11.refs,smiles_no_stereo/chunk_0.zst,idx/chunk_0.frames,11,2
```

`ZINC(...)` reads the second format.

## Build a ZINC Database

### 1. Download Chunks

Take a `.curl` file from <https://cartblanche.docking.org/tranches/2d> and place
it in `parser/`, for example `parser/ZINC22-downloader-small.gz.curl`.

The recommended path is to use `download_zinc`, because it downloads the files,
filters consecutive duplicate `ZINC ID`, writes `.gz` chunks, and creates the
required `chunk_table.csv`.

```bash
./parser/download_zinc \
  --smi-dir smiles \
  --curl-script parser/ZINC22-downloader-small.gz.curl \
  --producers 4
```

Use `--producers 4` as the default here; the upstream server seems only allows four
concurrent downloads.

If you already have raw chunks, create a `chunk_table.csv`
with the `chunk_id,abs_path,rel_path` schema shown above before continuing.

### 2. Remove Stereochemistry

`remove_stereo` reads the raw chunk table, removes stereochemistry with RDKit,
drops invalid molecules, filters consecutive duplicate ZINC IDs, and deduplicates
non-isomeric SMILES within each chunk. It accepts plain text, `.gz`, and `.bgzf`
inputs and writes `.zst` chunks plus a new `chunk_table.csv` into `--out-dir`.

```bash
./parser/remove_stereo \
  --chunk-table smiles/chunk_table.csv \
  --out-dir smiles_no_stereo \
  --producers 4 \
  --inner-threads 12
```

The next step must use `smiles_no_stereo/chunk_table.csv`, not the original
`smiles/chunk_table.csv`.

### 3. Generate Fingerprints and Search Manifests

`generate_fingerprints` reads the processed chunk table and writes:

- packed fingerprint arrays under `fp/fp_<size>/`
- reference indices under `idx/idx_<size>/`
- frame-offset tables under `idx/`
- Python manifests such as `tables/table_128.csv`

```bash
./parser/generate_fingerprints \
  --chunk-table smiles_no_stereo/chunk_table.csv \
  --fp-dir fp \
  --idx-dir idx \
  --table-dir tables \
  --fp-sizes 128 \
  -w 2
```

By default it generates sizes `64,128,256,512`. Any positive size divisible by
8 is accepted.

If you want to follow the Python example above, make sure `128` is included in
`--fp-sizes`, because the manifest filename is `tables/table_<fp_size>.csv`.

### Tests That Reproduce the Pipeline

The repository includes a small ZINC download script at
`parser/data/small_subset_download.curl` that can be used to exercise the full
download -> stereochemistry removal -> fingerprint generation pipeline.

The commands below write into `tmp/pipeline_test/` so they do not collide with
the generated files already checked into `parser/data/`.

```bash
mkdir -p tmp/pipeline_test

./parser/download_zinc \
  --smi-dir tmp/pipeline_test/raw \
  --curl-script parser/data/small_subset_download.curl \
  --producers 4

./parser/remove_stereo \
  --chunk-table tmp/pipeline_test/raw/chunk_table.csv \
  --out-dir tmp/pipeline_test/no_stereo \
  --producers 4 \
  --inner-threads 12

./parser/generate_fingerprints \
  --chunk-table tmp/pipeline_test/no_stereo/chunk_table.csv \
  --fp-dir tmp/pipeline_test/fp \
  --idx-dir tmp/pipeline_test/idx \
  --table-dir tmp/pipeline_test/tables \
  --fp-sizes 128 \
  -w 2
```

The resulting manifest is:

```text
tmp/pipeline_test/tables/table_128.csv
```

`test/tests.py` will use that manifest automatically if it exists. You can also
point the test suite at any generated manifest with:

```bash
CHEMICAL_SCREENER_TEST_TABLE=/path/to/table_128.csv pixi run python -m pytest test/tests.py -q
```

## Python API

After running the full pipeline, open the generated manifest and query it from
Python:

```python
import numpy as np

from chemical_screener.database import ZINC
from chemical_screener.run import get_sim_compounds, get_sim_scores

db = ZINC("tables/table_128.csv")

# Example single-query input: unpacked 0/1 bit vector of length 128.
fp = np.array([1, 0, 1, 0] * 32, dtype=np.uint8)

scores = get_sim_scores(x_b=db, x_q=fp)

hits = get_sim_compounds(
    x_b=db,
    x_q=fp,
    lower_bound=0.8,
    upper_bound=1.0,
)

hits_with_smiles = get_sim_compounds(
    x_b=db,
    x_q=fp,
    lower_bound=0.8,
    upper_bound=1.0,
    return_smiles=True,
)

fps = np.stack([fp, fp], axis=0)
multi_hits = get_sim_compounds(
    x_b=db,
    x_q=fps,
    lower_bound=0.8,
    upper_bound=1.0,
)
```

Return shapes:

- single-query `get_sim_compounds`: `{chunk_idx: {"score": ..., "positions" or "smiles": ...}}`
- multi-query `get_sim_compounds`: `{chunk_idx: {query_idx: {"score": ..., "positions" or "smiles": ...}}}`

## Enamine

`remove_stereo_enamine` is the equivalent preprocessing step for Enamine-style
delimited tables. It:

- reads chunk paths from `chunk_table.csv`
- detects whether the file is comma- or tab-delimited
- locates the `smiles` and `id` columns from the header
- removes stereochemistry from the `smiles` column
- writes normalized headerless `.zst` rows in `smiles<TAB>id` format
- writes a new `chunk_table.csv`

It accepts plain text and `.bz2` inputs.

```bash
./parser/remove_stereo_enamine \
  --chunk-table enamine/chunk_table.csv \
  --out-dir enamine_no_stereo \
  --producers 4 \
  --inner-threads 12
```

After that, run `generate_fingerprints` against
`enamine_no_stereo/chunk_table.csv` in the same way as for ZINC.

And use it the same way
```python
import numpy as np

from chemical_screener.database import DataBase
from chemical_screener.run import get_sim_compounds, get_sim_scores

db = DataBase("tables/table_128.csv")

# Example single-query input: unpacked 0/1 bit vector of length 128.
fp = np.array([1, 0, 1, 0] * 32, dtype=np.uint8)

scores = get_sim_scores(x_b=db, x_q=fp)
```
