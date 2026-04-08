import io
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple, Union
import os

from pathlib import Path

import numpy as np

import pandas as pd

try:
    import zstandard as zstd
except ImportError:  # Optional dependency for .zst-backed tables.
    zstd = None

# generate_fingerprints.cpp stores refs as packed uint32:
# (frame_id << 19) | row_in_frame.
_REF_FRAME_SHIFT = 19
_REF_BLOCK_OFFSET_MASK = (1 << _REF_FRAME_SHIFT) - 1

from chemical_screener.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "ZINC",
    "DataBase",
    "extract_bits_from_name",
    "get_smiles_and_ids_by_ref_indices",
]


def extract_bits_from_name(chunk_name: str) -> int:
    """Parse the integer bit count from a chunk identifier."""
    stem = Path(str(chunk_name)).stem
    _, sep, bit_count = stem.rpartition("_onbits_")
    if not sep or not bit_count.isdigit():
        raise ValueError(f"Could not parse on-bits value from {chunk_name!r}.")
    return int(bit_count)


class ZINC:
    def __init__(
        self,
        chunk_table: Union[str, os.PathLike],
        fp_size=128,
        dtype=np.uint8,
        name="zinc",
        max_files = None,
        **kwargs,
    ):
        df = pd.read_csv(chunk_table)
        if max_files is not None:
            df = df.iloc[:max_files, :]

        self.fp_files = df["npy"].values
        self.ref_files = df["refs"].values
        self.sizes = df["size"].values
        self.onbits = df["onbits"].values
        self.frames = df["frames"].values
        self.zst_files = df["zst"].values

        self.chunk_info = self._prepare_chunk_info()
        self._length_cache = sum(self.sizes)
        self.fp_size = fp_size

    @staticmethod
    def extract_bits_from_name(chunk_name: str) -> int:
        return extract_bits_from_name(chunk_name)

    def _prepare_chunk_info(
        self,
    ) -> List[Tuple[int, str, int, int, int, int]]:

        arguments = []
        start = 0
        for i in range(len(self.fp_files)):

            size = self.sizes[i]
            onbits = self.onbits[i]
            fp_file = self.fp_files[i]

            end = start + size

            arg = (i, fp_file, size, onbits, start, end)
            arguments.append(arg)

            start += size

        return arguments

    def get_smiles_and_ids_by_ref_indices(
        self,
        chunk_to_indices: dict[int, Sequence[int]],
        encoding: str = "utf-8",
    ) -> dict[int, List[Tuple[str, str]]]:
        """
        Fetch (smiles, zinc_id) pairs using packed uint32 refs into .zst blocks.

        Args:
            chunk_to_indices: {refs_chunk_index: [row_indices_in_refs]}
            encoding: text encoding for CSV rows.
        """
        if not chunk_to_indices:
            return {}

        if zstd is None:
            raise ImportError("zstandard is required to read .zst-backed chunk tables.")

        ref_files = self.ref_files
        frame_files = self.frames
        results: dict[int, List[Tuple[str, str]]] = {}
        dctx = zstd.ZstdDecompressor()
        for chunk_idx, indices in chunk_to_indices.items():
            chunk_idx = int(chunk_idx)
            ref_path = Path(ref_files[chunk_idx])
            frame_path = Path(frame_files[chunk_idx])
            if len(indices) == 0:
                results[chunk_idx] = []
                continue

            refs_map = np.memmap(ref_path, dtype="<u4", mode="r")
            total_refs = refs_map.shape[0]
            idx_array = np.asarray(indices, dtype=np.int64)
            if idx_array.ndim != 1:
                idx_array = idx_array.ravel()
            idx_array[idx_array < 0] += total_refs
            if np.any((idx_array < 0) | (idx_array >= total_refs)):
                raise IndexError(f"Ref index out of range for {ref_path}.")
            ref_values = np.asarray(refs_map[idx_array], dtype=np.uint32)

            rows: List[Optional[Tuple[str, str]]] = [None] * len(ref_values)
            by_frame: dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for pos, ref_value in enumerate(ref_values):
                frame_idx = int(ref_value >> _REF_FRAME_SHIFT)
                row_idx = int(ref_value & _REF_BLOCK_OFFSET_MASK)
                by_frame[frame_idx].append((pos, row_idx))

            frame_map = np.memmap(frame_path, dtype="<u8", mode="r")
            total_frames = frame_map.shape[0]
            zst_path = Path(self.zst_files[chunk_idx])
            zst_size = zst_path.stat().st_size

            with open(zst_path, "rb") as zst_file:
                for frame_idx, positions in by_frame.items():
                    if frame_idx < 0 or frame_idx >= total_frames:
                        raise IndexError(
                            f"Frame index {frame_idx} out of range for {frame_path}."
                        )
                    start = int(frame_map[frame_idx])
                    end = (
                        int(frame_map[frame_idx + 1])
                        if frame_idx + 1 < total_frames
                        else zst_size
                    )
                    zst_file.seek(start)
                    compressed = zst_file.read(end - start)
                    if (
                        zstd.get_frame_parameters(compressed).content_size
                        != zstd.CONTENTSIZE_UNKNOWN
                    ):
                        block = dctx.decompress(compressed)
                    else:
                        with dctx.stream_reader(io.BytesIO(compressed)) as reader:
                            block = reader.read()
                    positions.sort(key=lambda item: item[1])
                    target_idx = 0
                    target_row = positions[target_idx][1]
                    row = 0
                    start_idx = 0
                    while True:
                        end_idx = block.find(b"\n", start_idx)
                        if end_idx == -1:
                            line = block[start_idx:]
                            at_eof = True
                        else:
                            line = block[start_idx:end_idx]
                            at_eof = False
                        if line.endswith(b"\r"):
                            line = line[:-1]
                        if row == target_row:
                            parts = line.split(b"\t")
                            if len(parts) < 2:
                                raise ValueError(
                                    f"Malformed row in {zst_path} at row {row}."
                                )
                            while True:
                                pos, row_idx = positions[target_idx]
                                if row_idx != row:
                                    break
                                rows[pos] = (
                                    parts[0].decode(encoding),
                                    parts[1].decode(encoding),
                                )
                                target_idx += 1
                                if target_idx >= len(positions):
                                    break
                            if target_idx >= len(positions):
                                break
                            target_row = positions[target_idx][1]
                        if at_eof:
                            break
                        row += 1
                        start_idx = end_idx + 1
                    if target_idx < len(positions):
                        missing_row = positions[target_idx][1]
                        raise IndexError(
                            f"Row index {missing_row} out of range for frame {frame_idx}."
                        )

            resolved_rows: List[Tuple[str, str]] = []
            for row in rows:
                if row is None:
                    raise RuntimeError(
                        f"Failed to resolve rows for refs in {ref_path}."
                    )
                resolved_rows.append(row)
            results[chunk_idx] = resolved_rows

        return results

    def __len__(self):
        if self._length_cache is None:
            self._length_cache = sum(self.sizes)

        return self._length_cache


class DataBase(ZINC):
    """Backward-compatible alias for ``ZINC``."""


def get_smiles_and_ids_by_ref_indices(
    database: "ZINC",
    chunk_to_indices: dict[int, Sequence[int]],
    encoding: str = "utf-8",
) -> dict[int, List[Tuple[str, str]]]:
    """Module-level wrapper for the corresponding ``ZINC`` instance method."""
    return database.get_smiles_and_ids_by_ref_indices(
        chunk_to_indices=chunk_to_indices,
        encoding=encoding,
    )
