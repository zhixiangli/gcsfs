import asyncio
from unittest import mock

import fsspec.asyn
import pytest

from gcsfs.prefetcher import BackgroundPrefetcher, RunningAverageTracker, _fast_slice


class MockFetcher:
    def __init__(self, data, fail_at_call=None, hang_at_call=None):
        self.data = data
        self.calls = []
        self.fail_at_call = fail_at_call
        self.hang_at_call = hang_at_call
        self.call_count = 0

    async def __call__(self, start, size, split_factor=1):
        self.call_count += 1
        self.calls.append({"start": start, "size": size, "split_factor": split_factor})

        await asyncio.sleep(0.001)

        if self.hang_at_call is not None and self.call_count >= self.hang_at_call:
            await asyncio.sleep(1000)

        if self.fail_at_call is not None and self.call_count >= self.fail_at_call:
            raise OSError("Simulated Network Timeout")

        return self.data[start : start + size]


def test_fast_slice_direct():
    src = b"0123456789"
    assert _fast_slice(src, 2, 4) == b"2345"
    assert _fast_slice(src, 5, 0) == b""
    assert _fast_slice(src, 0, 10) == b"0123456789"


def test_running_average_tracker():
    tracker = RunningAverageTracker(maxlen=3)
    assert tracker.average == 1024 * 1024  # Default 1MB fallback

    tracker.add(512)
    tracker.add(512)
    assert tracker.average == 512

    tracker.add(2048)
    assert tracker.average == 1024  # (512 + 512 + 2048) // 3

    tracker.clear()
    assert tracker.average == 1024 * 1024


def test_max_prefetch_size_property():
    bp1 = BackgroundPrefetcher(fetcher=MockFetcher(b""), size=10000, concurrency=4)
    assert bp1.producer.max_prefetch_size == bp1.producer.MIN_PREFETCH_SIZE
    bp1.close()

    bp2 = BackgroundPrefetcher(fetcher=MockFetcher(b""), size=1000000000, concurrency=4)
    # Give it a history so it calculates 2x the io_size
    bp2.read_tracker.add(100 * 1024 * 1024)
    assert bp2.producer.max_prefetch_size == 200 * 1024 * 1024
    bp2.close()


def test_sequential_read_spanning_blocks():
    data = b"A" * 100 + b"B" * 100 + b"C" * 100
    fetcher = MockFetcher(data)
    bp = BackgroundPrefetcher(fetcher=fetcher, size=300, concurrency=4)
    bp.read_tracker.add(100)  # Seed the adaptive tracker

    assert bp._fetch(0, 100) == b"A" * 100
    assert bp._fetch(100, 150) == b"B" * 50
    assert bp.consumer._current_block_idx == 50
    assert bp._fetch(150, 250) == b"B" * 50 + b"C" * 50
    assert bp._fetch(250, 300) == b"C" * 50
    assert bp._fetch(300, 310) == b""

    bp.close()


def test_fetch_default_args_and_out_of_bounds():
    fetcher = MockFetcher(b"12345")
    bp = BackgroundPrefetcher(fetcher=fetcher, size=5, concurrency=4)

    assert bp._fetch(None, None) == b"12345"
    assert bp._fetch(None, 2) == b"12"
    assert bp._fetch(5, 10) == b""
    assert bp._fetch(10, 20) == b""
    assert bp._fetch(2, 2) == b""
    assert bp._fetch(4, 2) == b""

    bp.close()


def test_seek_logic():
    data = b"0123456789" * 10
    fetcher = MockFetcher(data)
    bp = BackgroundPrefetcher(fetcher=fetcher, size=100, concurrency=4)

    assert bp._fetch(0, 10) == data[0:10]
    assert bp._fetch(10, 20) == data[10:20]
    assert bp.user_offset == 20
    assert bp._fetch(50, 60) == data[50:60]
    assert bp.user_offset == 60
    assert bp._fetch(10, 20) == data[10:20]
    assert bp.user_offset == 20

    bp.close()


def test_exception_placed_in_queue():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)

    async def inject_error():
        await bp.queue.put(ValueError("Injected Producer Error"))

    fsspec.asyn.sync(bp.loop, inject_error)

    with pytest.raises(ValueError, match="Injected Producer Error"):
        bp._fetch(0, 50)

    assert isinstance(bp._error, ValueError)
    bp.close()


def test_producer_concurrency_streak_and_min_chunk():
    data = b"X" * 1000
    fetcher = MockFetcher(data)

    bp = BackgroundPrefetcher(fetcher=fetcher, size=1000, concurrency=4)
    bp.read_tracker.add(50)

    # Temporarily lower chunk limit for test
    original_min_chunk = bp.producer.MIN_CHUNK_SIZE
    bp.producer.MIN_CHUNK_SIZE = 10

    bp._fetch(0, 50)
    bp._fetch(50, 100)
    bp._fetch(100, 150)

    fsspec.asyn.sync(bp.loop, asyncio.sleep, 0.1)

    split_factors = [call["split_factor"] for call in fetcher.calls]
    assert split_factors[0] == 4
    assert max(split_factors) > 1
    assert max(split_factors) <= 4

    bp.producer.MIN_CHUNK_SIZE = original_min_chunk
    bp.close()


def test_producer_loop_space_constraints():
    data = b"Y" * 100
    fetcher = MockFetcher(data)

    bp = BackgroundPrefetcher(fetcher=fetcher, size=100, concurrency=4)
    bp.read_tracker.add(60)

    original_min_chunk = bp.producer.MIN_CHUNK_SIZE
    bp.producer.MIN_CHUNK_SIZE = 200

    assert bp._fetch(0, 10) == b"Y" * 10

    fsspec.asyn.sync(bp.loop, asyncio.sleep, 0.1)
    sizes = [call["size"] for call in fetcher.calls]
    assert all(s <= 100 for s in sizes)

    bp.producer.MIN_CHUNK_SIZE = original_min_chunk
    bp.close()


def test_producer_error_propagation():
    fetcher = MockFetcher(b"A" * 1000, fail_at_call=3)
    bp = BackgroundPrefetcher(fetcher=fetcher, size=1000, concurrency=4)
    bp.read_tracker.add(100)

    assert bp._fetch(0, 100) == b"A" * 100

    with pytest.raises(OSError, match="Simulated Network Timeout"):
        bp._fetch(100, 500)

    assert bp.is_stopped is True
    bp.close()


def test_read_after_close_or_error():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)
    bp.close()

    assert bp.is_stopped is True
    with pytest.raises(RuntimeError, match="The file instance has been closed"):
        bp._fetch(0, 10)

    bp2 = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)
    bp2._error = ValueError("Pre-existing error")
    with pytest.raises(ValueError, match="Pre-existing error"):
        bp2._fetch(0, 10)
    bp2.close()


def test_empty_queue_when_stopped():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 500), size=500, concurrency=4)
    bp.is_stopped = True

    with pytest.raises(RuntimeError, match="The file instance has been closed"):
        bp._fetch(0, 100)

    bp.close()


def test_cancel_all_tasks_cleans_queue_with_exceptions():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)

    async def inject_task():
        async def dummy_exception_task():
            raise ValueError("Hidden error")

        task = asyncio.create_task(dummy_exception_task())
        await bp.queue.put(task)
        await asyncio.sleep(0.05)

    fsspec.asyn.sync(bp.loop, inject_task)
    bp.close()
    assert bp.queue.empty()


def test_cleanup_cancels_active_tasks():
    bp = BackgroundPrefetcher(
        fetcher=MockFetcher(b"Z" * 1000), size=1000, concurrency=4
    )

    async def inject_task():
        async def dummy_task():
            await asyncio.sleep(3)

        task = asyncio.create_task(dummy_task())
        bp.producer._active_tasks.add(task)

    fsspec.asyn.sync(bp.loop, inject_task)

    assert len(bp.producer._active_tasks) > 0
    assert bp.is_stopped is False

    bp.close()

    assert bp.is_stopped is True
    assert len(bp.producer._active_tasks) == 0


def test_read_task_cancellation():
    bp = BackgroundPrefetcher(
        fetcher=MockFetcher(b"X" * 1000), size=1000, concurrency=4
    )

    async def inject_and_read():
        bp.is_stopped = True
        while not bp.queue.empty():
            bp.queue.get_nowait()

        cancel_task = asyncio.create_task(asyncio.sleep(10))
        cancel_task.cancel()
        await bp.queue.put(cancel_task)

        with pytest.raises(asyncio.CancelledError):
            await bp.consumer.consume(10)

    fsspec.asyn.sync(bp.loop, inject_and_read)
    bp.close()


def test_async_fetch_exception_trapping():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)

    def bad_sync(*args, **kwargs):
        raise RuntimeError("Simulated sync crash")

    with mock.patch("fsspec.asyn.sync", side_effect=bad_sync):
        with pytest.raises(RuntimeError, match="Simulated sync crash"):
            bp._fetch(0, 10)

    assert bp.is_stopped is True
    assert isinstance(bp._error, RuntimeError)
    bp.close()


def test_read_past_eof_internal():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 50), size=50, concurrency=4)
    bp.user_offset = 50
    res = bp._fetch(50, 60)
    assert res == b""
    bp.close()


def test_fetch_with_exact_block_matches():
    data = b"X" * 100
    bp = BackgroundPrefetcher(fetcher=MockFetcher(data), size=100, concurrency=4)
    bp.read_tracker.add(50)

    assert bp._fetch(0, 50) == b"X" * 50
    assert bp.consumer._current_block_idx == 50
    assert bp._fetch(50, 100) == b"X" * 50

    bp.close()


def test_queue_empty_race_condition():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)

    async def inject():
        bp.queue.put_nowait(asyncio.create_task(asyncio.sleep(0)))
        with mock.patch.object(bp.queue, "get_nowait", side_effect=asyncio.QueueEmpty):
            await bp.producer.stop()

    fsspec.asyn.sync(bp.loop, inject)
    bp.close()


def test_producer_space_remaining_break():
    bp = BackgroundPrefetcher(
        fetcher=MockFetcher(b"X" * 1000),
        size=1000,
        concurrency=4,
        max_prefetch_size=150,
    )
    bp._fetch(0, 10)
    fsspec.asyn.sync(bp.loop, asyncio.sleep, 0.1)
    bp.close()


def test_producer_min_chunk_logic():
    bp1 = BackgroundPrefetcher(
        fetcher=MockFetcher(b"X" * 1000),
        size=1000,
        concurrency=4,
        max_prefetch_size=300,
    )
    bp1.producer.MIN_CHUNK_SIZE = 100

    fsspec.asyn.sync(bp1.loop, asyncio.sleep, 0.1)
    bp1.close()

    bp2 = BackgroundPrefetcher(
        fetcher=MockFetcher(b"X" * 1000),
        size=1000,
        concurrency=4,
        max_prefetch_size=150,
    )
    bp2.producer.MIN_CHUNK_SIZE = 100
    fsspec.asyn.sync(bp2.loop, asyncio.sleep, 0.1)
    bp2.close()


def test_producer_loop_exception():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b""), size=100, concurrency=4)
    error_object = ValueError("Producer crash")
    bp.producer.get_io_size = mock.Mock(side_effect=error_object)

    with pytest.raises(ValueError, match="Producer crash"):
        bp._fetch(0, 10)

    assert bp.is_stopped is True
    assert bp._error == error_object

    with pytest.raises(ValueError, match="Producer crash"):
        bp._fetch(0, 10)
    bp.close()


def test_seek_same_offset():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b""), size=100, concurrency=4)
    fsspec.asyn.sync(bp.loop, bp._async_fetch, 0, 10)
    bp.close()


def test_read_history_maxlen():
    bp = BackgroundPrefetcher(
        fetcher=MockFetcher(b"X" * 2000), size=2000, concurrency=4
    )
    for i in range(12):
        bp._fetch(i * 10, (i + 1) * 10)
    assert len(bp.read_tracker._history) == 10
    bp.close()


def test_fast_slice_branch():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 200), size=200, concurrency=4)
    assert bp._fetch(0, 10) == b"X" * 10
    assert bp._fetch(10, 20) == b"X" * 10
    bp.close()


def test_fetch_stopped_during_execution():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)

    async def fake_async_fetch(start, end):
        bp.is_stopped = True
        return b"fake"

    with mock.patch.object(bp, "_async_fetch", new=fake_async_fetch):
        with pytest.raises(RuntimeError, match="The file instance has been closed"):
            bp._fetch(0, 10)
    bp.close()


def test_async_fetch_not_block_break():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b""), size=100, concurrency=4)

    async def fake_consume(size):
        return b""

    bp.consumer.consume = fake_consume
    bp.user_offset = 0

    res = fsspec.asyn.sync(bp.loop, bp._async_fetch, 0, 50)
    assert res == b""
    bp.close()


def test_fetch_stopped_before_execution():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)
    bp.is_stopped = True
    bp._error = None

    with pytest.raises(RuntimeError, match="The file instance has been closed"):
        bp._fetch(0, 10)
    bp.close()


def test_async_fetch_zero_copy_remainder():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X"), size=100, concurrency=4)
    bp.consumer._current_block = b"ABCDE"
    bp.consumer._current_block_idx = 0
    bp.user_offset = 0
    res = fsspec.asyn.sync(bp.loop, bp._async_fetch, 0, 5)
    assert res == b"ABCDE"
    assert bp.consumer._current_block_idx == 5
    bp.close()


def test_read_runtime_error_on_stopped_empty():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X"), size=100, concurrency=4)
    bp.is_stopped = True
    bp.producer.is_stopped = True

    while not bp.queue.empty():
        bp.queue.get_nowait()

    res = fsspec.asyn.sync(bp.loop, bp.consumer.consume, 10)
    assert res == b""
    bp.close()


def test_init_invalid_max_prefetch_size():
    with pytest.raises(
        ValueError,
        match=r"max_prefetch_size should be a positive integer",
    ):
        BackgroundPrefetcher(
            fetcher=MockFetcher(b""), size=1000, concurrency=4, max_prefetch_size=0
        )


def test_init_valid_max_prefetch_size_edge_case():
    bp = BackgroundPrefetcher(
        fetcher=MockFetcher(b""), size=1000, concurrency=4, max_prefetch_size=100
    )
    assert bp.producer._user_max_prefetch_size == 100
    bp.close()


def test_consumer_zero_size_checks():
    bp = BackgroundPrefetcher(fetcher=MockFetcher(b"X" * 100), size=100, concurrency=4)

    # 1. Test consume size <= 0
    res_consume_zero = fsspec.asyn.sync(bp.loop, bp.consumer.consume, 0)
    assert res_consume_zero == b""
    res_consume_neg = fsspec.asyn.sync(bp.loop, bp.consumer.consume, -5)
    assert res_consume_neg == b""

    # 2. Test _advance size <= 0 directly
    # (consume catches it early, so we call _advance directly to hit its internal check)
    res_advance_zero = fsspec.asyn.sync(
        bp.loop, bp.consumer._advance, 0, save_data=True
    )
    assert res_advance_zero == []
    res_advance_neg = fsspec.asyn.sync(
        bp.loop, bp.consumer._advance, -10, save_data=False
    )
    assert res_advance_neg == []

    bp.close()


def test_producer_min_chunk_inner_break():
    fetcher = MockFetcher(b"X" * 1000)
    bp = BackgroundPrefetcher(
        fetcher=fetcher, size=1000, concurrency=4, max_prefetch_size=400
    )

    bp.read_tracker.add(100)

    original_min_chunk = bp.producer.MIN_CHUNK_SIZE
    bp.producer.MIN_CHUNK_SIZE = 200

    async def trigger_loop():
        bp.producer.current_offset = 250
        bp.consumer.offset = 0
        bp.consumer.sequential_streak = 3  # makes prefetch_size = (3+1) * 100 = 400
        bp.wakeup_event.set()
        await asyncio.sleep(0.05)

    fsspec.asyn.sync(bp.loop, trigger_loop)

    assert fetcher.call_count == 0

    bp.producer.MIN_CHUNK_SIZE = original_min_chunk
    bp.close()


def test_producer_loop_break_on_stopped_after_wakeup():
    fetcher = MockFetcher(b"X" * 1000)
    bp = BackgroundPrefetcher(fetcher=fetcher, size=1000, concurrency=4)

    async def trigger_stop_and_wake():
        bp.producer.is_stopped = True
        bp.wakeup_event.set()
        await asyncio.sleep(0.05)

    fsspec.asyn.sync(bp.loop, trigger_stop_and_wake)

    # Verify the producer gracefully exited without doing work
    assert fetcher.call_count == 0
    bp.close()

def test_prefetcher_short_read_hang():
    class ShortReadFetcher:
        def __init__(self, data):
            self.data = data
            self.call_count = 0

        async def __call__(self, start, size, split_factor=1):
            self.call_count += 1
            # Only return half of the request, simulating a short read
            actual_size = max(1, size // 2)
            end = start + actual_size
            return self.data[start:end]

    data = b"X" * 100
    bp = BackgroundPrefetcher(fetcher=ShortReadFetcher(data), size=100, concurrency=1)

    # Normally it might hang here if the producer reached EOF but the consumer
    # still needs more bytes to satisfy the 100 bytes request.
    try:
        res = bp._fetch(0, 100)
    finally:
        bp.close()

    # The length of res should be exactly the bytes fetched.
    # It won't be 100 because of the short read and EOF breaking,
    # but the crucial thing is it doesn't hang!
    assert len(res) < 100
