"""Routes WebDataset gs:// reads through gcsfs via environment configuration."""

import io
import logging
import os
import time

READAHEAD_BLOCK_SIZE = 32 * 2**20

READ_MODES = ("default", "readahead_32mb", "whole_object")
READ_MODE_ENV = "GCSFS_SUBSYSTEM_WDS_READ_MODE"

# Default gcsfs concurrency passed explicitly per call.
DEFAULT_READ_CONCURRENCY = 4
READ_CONCURRENCY_ENV = "GCSFS_SUBSYSTEM_WDS_READ_CONCURRENCY"

# Client-side buffer size in bytes (0 for none).
DEFAULT_READ_BUFFER_BYTES = 0
READ_BUFFER_ENV = "GCSFS_SUBSYSTEM_WDS_READ_BUFFER"


def _fs():
    import gcsfs

    return gcsfs.GCSFileSystem()


def current_read_mode():
    return os.environ.get(READ_MODE_ENV, "default")


def current_read_concurrency():
    return int(os.environ.get(READ_CONCURRENCY_ENV, DEFAULT_READ_CONCURRENCY))


def current_read_buffer_bytes():
    return int(os.environ.get(READ_BUFFER_ENV, DEFAULT_READ_BUFFER_BYTES))


def open_url(
    url,
    mode="rb",
    read_mode="default",
    concurrency=DEFAULT_READ_CONCURRENCY,
    buffer_bytes=DEFAULT_READ_BUFFER_BYTES,
    fs=None,
):
    """Opens a gs:// URL through gcsfs using the selected read mode."""
    if mode != "rb":
        raise ValueError(f"gcsfs opener only supports mode 'rb', got {mode!r}")
    fs = fs if fs is not None else _fs()
    if read_mode == "whole_object":
        # Whole object is already in memory and cannot be buffered.
        if buffer_bytes:
            raise ValueError("whole_object reads cannot take a read buffer")
        return io.BytesIO(fs.cat_file(url, concurrency=concurrency))
    if read_mode == "readahead_32mb":
        # Explicit cache_type disables gcsfs adaptive prefetcher.
        return _buffered(
            fs.open(
                url,
                "rb",
                block_size=READAHEAD_BLOCK_SIZE,
                cache_type="readahead",
                concurrency=concurrency,
            ),
            buffer_bytes,
        )
    if read_mode == "default":
        # Omit cache_type to retain gcsfs adaptive prefetcher.
        return _buffered(fs.open(url, "rb", concurrency=concurrency), buffer_bytes)
    raise ValueError(f"unknown read_mode {read_mode!r}; expected one of {READ_MODES}")


def _buffered(handle, buffer_bytes):
    """Wraps handle in io.BufferedReader to coalesce small reads."""
    if not buffer_bytes:
        return handle
    return io.BufferedReader(handle, buffer_size=buffer_bytes)


# A shard slower than this is logged: at ~95 MB/shard even 10 MB/s finishes in time.
SLOW_SHARD_SECONDS = 10.0


def _now():
    return time.monotonic()


class _TimedShard:
    """Read-through proxy that logs a shard whose open-to-last-read time is slow.

    WebDataset drops the stream without closing it, so the report fires on close()
    or garbage collection, whichever comes first, and times up to the last read.
    """

    def __init__(self, stream, url, read_mode, start):
        self.stream = stream
        self._url = url
        self._mode = read_mode
        self._start = start
        self._first = None
        self._last = start
        self._bytes = 0
        self._max_read = 0.0
        self._reported = False

    def read(self, *args):
        begin = _now()
        data = self.stream.read(*args)
        end = _now()
        if self._first is None:
            self._first = begin
        self._last = end
        self._bytes += len(data)
        self._max_read = max(self._max_read, end - begin)
        return data

    def _report(self):
        if self._reported:
            return
        self._reported = True
        seconds = self._last - self._start
        if seconds < SLOW_SHARD_SECONDS:
            return
        first = self._first if self._first is not None else self._last
        print(
            f"slow_shard url={self._url} mode={self._mode} seconds={seconds:.2f} "
            f"open_seconds={first - self._start:.2f} bytes={self._bytes} "
            f"max_read_seconds={self._max_read:.2f} pid={os.getpid()}",
            flush=True,
        )

    def close(self):
        self._report()
        self.stream.close()

    def __del__(self):
        try:
            self._report()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def gopen_gcsfs(url, mode="rb", bufsize=8192, **kw):
    """WebDataset gopen handler for gs:// URLs (ignores pipe bufsize hint)."""
    read_mode = current_read_mode()
    start = _now()
    stream = open_url(
        url,
        mode,
        read_mode=read_mode,
        concurrency=current_read_concurrency(),
        buffer_bytes=current_read_buffer_bytes(),
    )
    return _TimedShard(stream, url, read_mode, start)


class _RetryPrinter(logging.Handler):
    """Prints gcsfs retry messages, which gcsfs only logs at DEBUG."""

    def filter(self, record):
        message = record.getMessage()
        return "retrying" in message or "out of retries" in message

    def emit(self, record):
        print(f"gcsfs_retry pid={record.process} {record.getMessage()}", flush=True)


def _surface_gcsfs_retries():
    logger = logging.getLogger("gcsfs")
    if any(isinstance(h, _RetryPrinter) for h in logger.handlers):
        return
    logger.addHandler(_RetryPrinter(level=logging.DEBUG))
    logger.setLevel(logging.DEBUG)


def register():
    import webdataset as wds

    wds.gopen_schemes["gs"] = gopen_gcsfs
    _surface_gcsfs_retries()


def is_registered():
    """Returns True if gs:// is registered to gopen_gcsfs in this process."""
    import webdataset as wds

    return wds.gopen_schemes.get("gs") is gopen_gcsfs


def worker_init(worker_id):
    """DataLoader worker_init_fn to register the gs:// handler in spawned workers."""
    del worker_id
    register()
