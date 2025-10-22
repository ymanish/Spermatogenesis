
import psutil
import os
import csv
from pathlib import Path

import tempfile
from src.utils.logger_util import get_logger
import logging


SCRATCH_PATH  = None
FLUSH_EVERY   = None
WORKER_LOGGER = None

def init_worker(flush_every: int = 10000):
    """Run once in every child before it executes any task."""

    global WORKER_LOGGER, SCRATCH_PATH, FLUSH_EVERY ### I need to define them global because I am changing the values in the function for these variables

    FLUSH_EVERY = flush_every
    proc = psutil.Process(os.getpid())
    WORKER_LOGGER = get_logger(f"Worker-{os.getpid()}", log_file=None, level=logging.INFO)


    scratch_tmpdir = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")

    if not scratch_tmpdir or not Path(scratch_tmpdir).is_dir():
        raise RuntimeError("Scratch directory not found")

    # tmpfile = tempfile.NamedTemporaryFile(mode="w", dir=scratch_tmpdir,
    #                                       prefix=f"wbatch-{os.getpid()}-", suffix=".tsv",
    #                                       delete=False, newline='')
    
    # SCRATCH_PATH = Path(tmpfile.name)
    # TSV_WRITER = csv.writer(tmpfile, delimiter="\t",
    #                         lineterminator="\n")

    SCRATCH_PATH = Path(scratch_tmpdir)
               
    WORKER_LOGGER.info("Worker %s online - RSS %.1f MB",
                 os.getpid(), proc.memory_info().rss / 2**20)
    WORKER_LOGGER.info("Temporary file created at %s", SCRATCH_PATH)    

    # DEBUG: show all four globals immediately after init
    # logger = get_logger("bkeep.init_worker")
    # logger.info(
    #     "[DEBUG][bkeep.init_worker] TSV_WRITER=%s, SCRATCH_PATH=%s, "
    #     "FLUSH_EVERY=%s, WORKER_LOGGER=%s",
    #     TSV_WRITER, SCRATCH_PATH, FLUSH_EVERY, WORKER_LOGGER
    # )

def new_batch_writer(fmt: str = "tsv", suffix: str = None):
   
    if SCRATCH_PATH is None:
        raise RuntimeError("SCRATCH_PATH not set; call init_worker first")

    pid = os.getpid()
    fmt = fmt.lower()
    if fmt == "tsv":
        suffix = suffix or ".tsv"
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            dir=SCRATCH_PATH,
            prefix=f"batch-{pid}-",
            suffix=suffix,
            delete=False,
            newline=""
        )
        writer = csv.writer(tmp, delimiter="\t", lineterminator="\n")
        return tmp, writer

    elif fmt in ("pkl", "bin"):
        suffix = suffix or ".pkl"
        tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=SCRATCH_PATH,
            prefix=f"batch-{pid}-",
            suffix=suffix,
            delete=False
        )
        return tmp, None
    
    elif fmt == "parquet":
        # create an empty file path for Parquet output
        suffix = suffix or ".parquet"
        fd, path = tempfile.mkstemp(
            prefix=f"batch-{pid}-",
            suffix=suffix,
            dir=SCRATCH_PATH
        )
        os.close(fd)
        return path

    else:
        raise ValueError(f"Unknown format {fmt!r}, expected 'tsv' or 'pkl'")