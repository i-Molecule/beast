from setuptools import setup, Command
from setuptools.command.build import build as _build
from pathlib import Path
import subprocess

class BuildShared(Command):
    description = "build libtanimoto.so with g++"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        src_dir = Path(__file__).parent / "src" / "chemical_screener"
        output_path = src_dir / "libtanimoto.so"
        cmd = [
            "g++",
            "-march=native",
            "-mtune=native",
            "-O3",
            "-fopenmp",
            "-fPIC",
            "-shared",
            "-o",
            str(output_path),
            str(src_dir / "tanimoto_multi_query_overlap.cpp"),
            str(src_dir / "tanimoto_multi_query_overlap_packed.cpp"),
            str(src_dir / "tanimoto_single_query.cpp"),
            str(src_dir / "tanimoto_single_query_fp16.cpp"),
            str(src_dir / "tanimoto_single_query_packed_fp16.cpp"),
            str(src_dir / "tanimoto_single_query_filter.cpp"),
            str(src_dir / "tanimoto_single_query_filter_packed.cpp"),
        ]
        self.announce("Building libtanimoto.so", level=2)
        subprocess.check_call(cmd)


class build(_build):
    sub_commands = [("build_shared", None)] + _build.sub_commands


setup(
    # Package metadata lives in pyproject.toml; setup.py only extends the build.
    cmdclass={"build_shared": BuildShared, "build": build},
    options={
        "build_ext": {
            "build_lib": "build/ext",
            "build_temp": "build/temp",
        }
    },
)
