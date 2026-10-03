import io

import pytest

from gcsfs.tests.perf.subsystembenchmarks.dataloading.webdataset import gcsfs_opener


def torch_dataset_base():
    """Returns torch IterableDataset if installed, else object."""
    try:
        from torch.utils.data import IterableDataset
    except ImportError:
        return object
    return IterableDataset


class FakeFS:
    """Mock filesystem recording open and cat_file calls."""

    def __init__(self):
        self.calls = []

    def open(self, url, mode, **kwargs):
        self.calls.append(("open", url, mode, kwargs))
        return io.BytesIO(b"payload")

    def cat_file(self, url, **kwargs):
        self.calls.append(("cat_file", url, kwargs))
        return b"payload"


def test_default_mode_passes_no_cache_type_or_block_size():
    """Default read mode leaves cache_type unset to preserve adaptive prefetcher."""
    fs = FakeFS()
    stream = gcsfs_opener.open_url("gs://b/s.tar", read_mode="default", fs=fs)
    assert stream.read() == b"payload"
    _, url, mode, kwargs = fs.calls[0]
    assert (url, mode) == ("gs://b/s.tar", "rb")
    assert kwargs == {"concurrency": 4}  # No cache_type: prefetcher remains active.


def test_readahead_mode_sets_block_size_and_cache_type():
    fs = FakeFS()
    gcsfs_opener.open_url("gs://b/s.tar", read_mode="readahead_32mb", fs=fs)
    _, _, _, kwargs = fs.calls[0]
    assert kwargs == {
        "block_size": 32 * 2**20,
        "cache_type": "readahead",
        "concurrency": 4,
    }


def test_whole_object_mode_uses_cat_file():
    fs = FakeFS()
    stream = gcsfs_opener.open_url(
        "gs://b/s.tar", read_mode="whole_object", concurrency=16, fs=fs
    )
    assert fs.calls == [("cat_file", "gs://b/s.tar", {"concurrency": 16})]
    assert stream.read() == b"payload"


def test_read_buffer_coalesces_small_reads_before_they_reach_gcsfs():
    """Verifies client-side buffer coalesces small reader chunks into larger GCS reads."""

    class RecordingFS:
        def __init__(self):
            self.reads = []

        def open(self, url, mode, **kwargs):
            fs = self

            class Handle(io.RawIOBase):
                def __init__(self):
                    self.off = 0

                def readable(self):
                    return True

                def readinto(self, buf):
                    n = min(len(buf), 1 << 20)
                    fs.reads.append(n)
                    buf[:n] = b"\0" * n
                    self.off += n
                    return n

                def read(self, size=-1):
                    fs.reads.append(size)
                    return b"\0" * size

            return Handle()

    plain = RecordingFS()
    stream = gcsfs_opener.open_url("gs://b/s.tar", fs=plain)
    for _ in range(8):
        stream.read(10240)
    assert plain.reads == [10240] * 8

    buffered = RecordingFS()
    stream = gcsfs_opener.open_url("gs://b/s.tar", buffer_bytes=1 << 20, fs=buffered)
    for _ in range(8):
        stream.read(10240)
    assert buffered.reads == [1 << 20], "8 tar-sized reads must cost one gcsfs read"


def test_whole_object_rejects_a_read_buffer():
    """whole_object mode is already in memory and cannot accept a read buffer."""
    with pytest.raises(ValueError, match="whole_object"):
        gcsfs_opener.open_url(
            "gs://b/s.tar", read_mode="whole_object", buffer_bytes=1 << 20, fs=FakeFS()
        )


def test_read_buffer_travels_from_the_environment(monkeypatch):
    fs = FakeFS()
    monkeypatch.setattr(gcsfs_opener, "_fs", lambda: fs)
    monkeypatch.setenv(gcsfs_opener.READ_MODE_ENV, "default")
    monkeypatch.setenv(gcsfs_opener.READ_BUFFER_ENV, str(4 << 20))
    stream = gcsfs_opener.gopen_gcsfs("gs://b/s.tar", "rb", 8192)
    assert isinstance(stream.stream, io.BufferedReader)


def test_unknown_read_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown read_mode"):
        gcsfs_opener.open_url("gs://b/s.tar", read_mode="nope", fs=FakeFS())


def test_write_mode_is_rejected():
    with pytest.raises(ValueError, match="only supports"):
        gcsfs_opener.open_url("gs://b/s.tar", mode="wb", fs=FakeFS())


def test_webdataset_bufsize_is_discarded(monkeypatch):
    """Ignores WebDataset's default 8192 pipe buffer hint in favor of block_size."""
    fs = FakeFS()
    monkeypatch.setattr(gcsfs_opener, "_fs", lambda: fs)
    monkeypatch.setenv(gcsfs_opener.READ_MODE_ENV, "readahead_32mb")
    gcsfs_opener.gopen_gcsfs("gs://b/s.tar", "rb", 8192)
    _, _, _, kwargs = fs.calls[0]
    assert 8192 not in kwargs.values()
    assert kwargs["block_size"] == 32 * 2**20


class _WorkerProbe(torch_dataset_base()):
    """Iterable dataset that yields environment and registration state in workers."""

    def __iter__(self):
        yield {
            "registered": gcsfs_opener.is_registered(),
            "read_mode": gcsfs_opener.current_read_mode(),
            "concurrency": gcsfs_opener.current_read_concurrency(),
            "buffer_bytes": gcsfs_opener.current_read_buffer_bytes(),
        }


def test_a_spawned_dataloader_worker_gets_the_handler_and_the_read_environment(
    monkeypatch,
):
    pytest.importorskip("webdataset")
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(gcsfs_opener.READ_MODE_ENV, "readahead_32mb")
    monkeypatch.setenv(gcsfs_opener.READ_CONCURRENCY_ENV, "16")
    monkeypatch.setenv(gcsfs_opener.READ_BUFFER_ENV, str(8 << 20))

    loader = torch.utils.data.DataLoader(
        _WorkerProbe(),
        num_workers=1,
        batch_size=None,
        worker_init_fn=gcsfs_opener.worker_init,
        multiprocessing_context="spawn",
    )
    seen = next(iter(loader))

    assert seen["registered"], "worker_init_fn did not reach the spawned worker"
    assert seen["read_mode"] == "readahead_32mb"
    assert int(seen["concurrency"]) == 16
    assert int(seen["buffer_bytes"]) == 8 << 20


def test_read_concurrency_travels_from_the_environment(monkeypatch):
    """Verifies read concurrency setting is read from the environment."""
    fs = FakeFS()
    monkeypatch.setattr(gcsfs_opener, "_fs", lambda: fs)
    monkeypatch.setenv(gcsfs_opener.READ_MODE_ENV, "default")
    monkeypatch.setenv(gcsfs_opener.READ_CONCURRENCY_ENV, "16")
    gcsfs_opener.gopen_gcsfs("gs://b/s.tar", "rb", 8192)
    assert fs.calls[0][3]["concurrency"] == 16


def _clock(monkeypatch, *ticks):
    it = iter(ticks)
    monkeypatch.setattr(gcsfs_opener, "_now", lambda: next(it))


def test_slow_shard_is_reported_with_its_url_and_longest_read(monkeypatch, capsys):
    fs = FakeFS()
    monkeypatch.setattr(gcsfs_opener, "_fs", lambda: fs)
    monkeypatch.setenv(gcsfs_opener.READ_MODE_ENV, "default")
    # open at 0; read() spans 1 -> 13.
    _clock(monkeypatch, 0.0, 1.0, 13.0)

    stream = gcsfs_opener.gopen_gcsfs("gs://b/s.tar", "rb", 8192)
    assert stream.read() == b"payload"
    stream.close()

    out = capsys.readouterr().out
    assert "slow_shard url=gs://b/s.tar mode=default seconds=13.00" in out
    assert "open_seconds=1.00 bytes=7 max_read_seconds=12.00" in out


def test_fast_shard_prints_nothing(monkeypatch, capsys):
    fs = FakeFS()
    monkeypatch.setattr(gcsfs_opener, "_fs", lambda: fs)
    monkeypatch.setenv(gcsfs_opener.READ_MODE_ENV, "whole_object")
    _clock(monkeypatch, 0.0, 0.5, 0.6)

    with gcsfs_opener.gopen_gcsfs("gs://b/s.tar", "rb", 8192) as stream:
        assert stream.read() == b"payload"

    assert "slow_shard" not in capsys.readouterr().out


def test_gcsfs_retries_are_surfaced_after_register(capsys):
    import logging

    pytest.importorskip("webdataset")
    gcsfs_opener.register()
    logger = logging.getLogger("gcsfs")
    logger.debug("GET: noise that must stay hidden")
    logger.debug("_cat_file retrying after exception: boom")

    out = capsys.readouterr().out
    assert "gcsfs_retry" in out and "boom" in out
    assert "noise" not in out


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_slow_request_is_broken_down_into_queue_connect_ttfb_and_body(
    monkeypatch, capsys
):
    from types import SimpleNamespace

    pytest.importorskip("aiohttp")
    trace = gcsfs_opener.slow_request_trace()
    ctx = SimpleNamespace()
    params = SimpleNamespace(
        method="GET", url="https://storage.googleapis.com/b/o", headers={"Range": "bytes=0-9"}
    )
    # start 0; queued 0 -> 1; connect 1 -> 1.5; headers at 6; body done at 9.
    _clock(monkeypatch, 0.0, 0.0, 1.0, 1.0, 1.5, 6.0, 9.0)
    _run(trace.on_request_start[0](None, ctx, params))
    _run(trace.on_connection_queued_start[0](None, ctx, None))
    _run(trace.on_connection_queued_end[0](None, ctx, None))
    _run(trace.on_connection_create_start[0](None, ctx, None))
    _run(trace.on_connection_create_end[0](None, ctx, None))
    _run(trace.on_request_end[0](None, ctx, params))
    _run(trace.on_response_chunk_received[0](None, ctx, params))

    out = capsys.readouterr().out
    assert (
        "slow_request method=GET range=bytes=0-9 seconds=9.00 queued=1.00 "
        "connect=0.50 ttfb=6.00 body=3.00"
    ) in out
    assert "port=-" in out


def test_slow_request_includes_local_port(monkeypatch, capsys):
    from types import SimpleNamespace

    pytest.importorskip("aiohttp")
    trace = gcsfs_opener.slow_request_trace()
    ctx = SimpleNamespace()
    params = SimpleNamespace(
        method="GET", url="https://storage.googleapis.com/b/o", headers={"Range": "bytes=0-9"}
    )
    _clock(monkeypatch, 0.0, 6.0, 9.0)
    _run(trace.on_request_start[0](None, ctx, params))
    ctx.lport = 54321
    _run(trace.on_request_end[0](None, ctx, params))
    _run(trace.on_response_chunk_received[0](None, ctx, params))

    out = capsys.readouterr().out
    assert "slow_request method=GET range=bytes=0-9" in out
    assert "port=54321" in out


def test_fast_request_prints_nothing(monkeypatch, capsys):
    from types import SimpleNamespace

    pytest.importorskip("aiohttp")
    trace = gcsfs_opener.slow_request_trace()
    ctx = SimpleNamespace()
    params = SimpleNamespace(method="GET", url="u", headers={})
    _clock(monkeypatch, 0.0, 0.1, 0.2)
    _run(trace.on_request_start[0](None, ctx, params))
    _run(trace.on_request_end[0](None, ctx, params))
    _run(trace.on_response_chunk_received[0](None, ctx, params))

    assert "slow_request" not in capsys.readouterr().out


def test_opener_filesystem_carries_the_trace_and_is_reused():
    pytest.importorskip("aiohttp")
    fs = gcsfs_opener._fs()
    assert fs.session_kwargs["trace_configs"] == [gcsfs_opener.slow_request_trace()]
    assert gcsfs_opener._fs() is fs


def test_port_tracker_wraps_connect_and_records_local_port(monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("aiohttp")
    import aiohttp

    class FakeTransport:
        def get_extra_info(self, name):
            return ("127.0.0.1", 49152) if name == "sockname" else None

    fake_conn = SimpleNamespace(transport=FakeTransport())
    calls = []

    async def fake_connect(self, req, traces, timeout):
        calls.append((self, req, traces, timeout))
        return fake_conn

    # Install the tracker on top of a fake original connect.
    monkeypatch.setattr(aiohttp.TCPConnector, "connect", fake_connect)
    monkeypatch.setattr(
        aiohttp.TCPConnector, "_port_tracking_installed", False, raising=False
    )
    gcsfs_opener._install_port_tracker()
    assert aiohttp.TCPConnector.connect is not fake_connect

    ctx = SimpleNamespace()
    traces = [SimpleNamespace(_trace_config_ctx=ctx)]
    conn = _run(aiohttp.TCPConnector.connect("self", "req", traces, "timeout"))

    assert conn is fake_conn
    assert calls == [("self", "req", traces, "timeout")]
    assert ctx.lport == 49152


def test_slow_request_reports_the_real_local_port_of_fresh_and_reused_connections(
    monkeypatch, capsys
):
    pytest.importorskip("aiohttp")
    import aiohttp
    from aiohttp import web

    monkeypatch.setattr(gcsfs_opener, "SLOW_REQUEST_SECONDS", 0.0)
    trace = gcsfs_opener.slow_request_trace()
    client_ports = []

    async def handler(request):
        client_ports.append(request.transport.get_extra_info("peername")[1])
        return web.Response(body=b"x" * 1024)

    async def _test():
        app = web.Application()
        app.router.add_get("/o", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession(trace_configs=[trace]) as session:
                for _ in range(2):
                    async with session.get(f"http://127.0.0.1:{port}/o") as resp:
                        await resp.read()
        finally:
            await runner.cleanup()

    _run(_test())

    out = capsys.readouterr().out
    # Second request reuses the pooled connection, so both share one local port.
    assert len(client_ports) == 2 and client_ports[0] == client_ports[1]
    lines = [line for line in out.splitlines() if line.startswith("slow_request ")]
    assert lines and all(f"port={client_ports[0]}" in line for line in lines)
    assert "port=-" not in out
