"""Routes WebDataset gs:// reads through gcsfs via environment configuration."""

import functools
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

    # One TraceConfig per process keeps fsspec's instance cache (and so one session).
    return gcsfs.GCSFileSystem(session_kwargs={"trace_configs": [slow_request_trace()]})


# A single GET slower than this is logged with its phase breakdown.
SLOW_REQUEST_SECONDS = 3.0


def _install_port_tracker():
    """Wrap TCPConnector.connect to attach the connection local port to trace context."""
    import aiohttp

    if getattr(aiohttp.TCPConnector, "_port_tracking_installed", False):
        return

    _orig_connect = aiohttp.TCPConnector.connect

    async def _tracking_connect(self, req, traces, timeout):
        conn = await _orig_connect(self, req, traces, timeout)
        try:
            sockname = conn.transport.get_extra_info("sockname") if conn.transport else None
            lport = sockname[1] if sockname and len(sockname) > 1 else None
            if traces and lport:
                for t in traces:
                    ctx = getattr(t, "_trace_config_ctx", None)
                    if ctx is not None:
                        ctx.lport = lport
        except Exception:
            pass
        return conn

    aiohttp.TCPConnector.connect = _tracking_connect
    aiohttp.TCPConnector._port_tracking_installed = True


@functools.lru_cache(maxsize=None)
def slow_request_trace():
    """aiohttp TraceConfig printing queue/connect/ttfb/body times of slow requests."""
    import aiohttp

    _install_port_tracker()
    trace = aiohttp.TraceConfig()

    async def request_start(session, ctx, params):
        ctx.start = _now()
        ctx.queued = ctx.connect = 0.0
        ctx.headers_at = None
        ctx.range = params.headers.get("Range", "-")
        ctx.lport = None

    async def queued_start(session, ctx, params):
        ctx.queued_from = _now()

    async def queued_end(session, ctx, params):
        ctx.queued += _now() - ctx.queued_from

    async def create_start(session, ctx, params):
        ctx.connect_from = _now()

    async def create_end(session, ctx, params):
        ctx.connect += _now() - ctx.connect_from

    async def request_end(session, ctx, params):
        ctx.headers_at = _now()
        if getattr(ctx, "lport", None) is None:
            try:
                proto = getattr(params.response, "_protocol", None)
                trans = getattr(proto, "transport", None) if proto else None
                sockname = trans.get_extra_info("sockname") if trans else None
                if sockname and len(sockname) > 1:
                    ctx.lport = sockname[1]
            except Exception:
                pass

    async def body_done(session, ctx, params):
        end = _now()
        if ctx.headers_at is None or end - ctx.start < SLOW_REQUEST_SECONDS:
            return
        port = getattr(ctx, "lport", None)
        port_str = f" port={port}" if port is not None else " port=-"
        print(
            f"slow_request method={params.method} range={ctx.range} "
            f"seconds={end - ctx.start:.2f} queued={ctx.queued:.2f} "
            f"connect={ctx.connect:.2f} ttfb={ctx.headers_at - ctx.start:.2f} "
            f"body={end - ctx.headers_at:.2f} pid={os.getpid()}{port_str}",
            flush=True,
        )

    async def request_exception(session, ctx, params):
        port = getattr(ctx, "lport", None)
        port_str = f" port={port}" if port is not None else " port=-"
        print(
            f"slow_request_error method={params.method} range={ctx.range} "
            f"seconds={_now() - ctx.start:.2f} error={params.exception!r} "
            f"pid={os.getpid()}{port_str}",
            flush=True,
        )

    trace.on_request_start.append(request_start)
    trace.on_connection_queued_start.append(queued_start)
    trace.on_connection_queued_end.append(queued_end)
    trace.on_connection_create_start.append(create_start)
    trace.on_connection_create_end.append(create_end)
    trace.on_request_end.append(request_end)
    trace.on_response_chunk_received.append(body_done)
    trace.on_request_exception.append(request_exception)
    return trace


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

    _install_port_tracker()
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
