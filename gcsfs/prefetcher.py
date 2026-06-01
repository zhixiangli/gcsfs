import asyncio
import ctypes
import logging
import threading
from collections import deque

import fsspec.asyn

logger = logging.getLogger(__name__)

PyBytes_FromStringAndSize = ctypes.pythonapi.PyBytes_FromStringAndSize
PyBytes_FromStringAndSize.restype = ctypes.py_object
PyBytes_FromStringAndSize.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]

PyBytes_AsString = ctypes.pythonapi.PyBytes_AsString
PyBytes_AsString.restype = ctypes.c_void_p
PyBytes_AsString.argtypes = [ctypes.py_object]


# Please refer to following discussion to understand why this is required at this point
# Discussion = https://github.com/fsspec/gcsfs/pull/795#discussion_r3032749881
def _fast_slice(src_bytes, offset, read_size):
    if read_size == 0:
        return b""
    dest_bytes = PyBytes_FromStringAndSize(None, read_size)
    src_ptr = PyBytes_AsString(src_bytes)
    dest_ptr = PyBytes_AsString(dest_bytes)

    ctypes.memmove(dest_ptr, src_ptr + offset, read_size)
    return dest_bytes


class RunningAverageTracker:
    """Tracks a running average of values over a sliding window.

    This is used to monitor read sizes and adaptively scale the
    prefetching strategy based on recent user behavior.
    """

    def __init__(self, maxlen=10):
        """Initializes the tracker with a specific window size.

        Args:
            maxlen (int): The maximum number of historical values to keep.
        """
        logger.debug("Initializing RunningAverageTracker with maxlen: %d", maxlen)
        self._history = deque(maxlen=maxlen)
        self._sum = 0

    def add(self, value: int):
        """Adds a new value to the sliding window and updates the rolling sum.

        Args:
            value (int): The integer value to add to the history.
        """
        if value <= 0:
            raise ValueError(
                "Internal error, RunningAverageTracker tried inserting negative value"
            )
        if len(self._history) == self._history.maxlen:
            self._sum -= self._history[0]

        self._history.append(value)
        self._sum += value
        logger.debug(
            "RunningAverageTracker added value: %d, new sum: %d", value, self._sum
        )

    @property
    def average(self) -> int:
        """Calculates and returns the current running average.

        Returns:
            int: The integer average of the current history.
        """
        count = len(self._history)
        if count == 0:
            return 1024 * 1024  # 1MB
        return self._sum // count

    def clear(self):
        """Clears the history and resets the sum to zero."""
        logger.debug("Clearing RunningAverageTracker history.")
        self._history.clear()
        self._sum = 0


class PrefetchProducer:
    """Background worker that fetches sequential blocks of data.

    This class handles the network requests. It spawns asynchronous tasks
    to fetch data ahead of the user's current reading position and
    places those task promises into a queue for the consumer.
    """

    # If the request is too small, and prefetch window is expanded till 5MB
    # we then make request in 5MB blocks.
    MIN_CHUNK_SIZE = 5 * 1024 * 1024

    # If user doesn't specify any max_prefetch_size, the prefetcher defaults
    # to maximum of 2 * io_size and 128MB
    MIN_PREFETCH_SIZE = 128 * 1024 * 1024

    def __init__(
        self,
        fetcher,
        size: int,
        concurrency: int,
        queue: asyncio.Queue,
        wakeup_event: asyncio.Event,
        get_user_offset,
        get_io_size,
        get_sequential_streak,
        on_error,
        user_max_prefetch_size=None,
    ):
        """Initializes the background producer.

        Args:
            fetcher (Callable): A coroutine function to fetch bytes from a remote source.
            size (int): Total size of the file being fetched.
            concurrency (int): Maximum number of concurrent fetch tasks.
            queue (asyncio.Queue): The shared queue to push download tasks into.
            wakeup_event (asyncio.Event): Event used to wake the producer from an idle state.
            get_user_offset (Callable): Function returning the user's current read offset.
            get_io_size (Callable): Function returning the adaptive IO size.
            get_sequential_streak (Callable): Function returning the current sequential read streak.
            on_error (Callable): Callback triggered when a background error occurs.
            user_max_prefetch_size (int, optional): A hard limit for prefetch size overrides.
        """
        logger.debug(
            "Initializing PrefetchProducer: size=%d, concurrency=%d, user_max_prefetch_size=%s",
            size,
            concurrency,
            user_max_prefetch_size,
        )
        self.fetcher = fetcher
        self.size = size
        self.concurrency = concurrency
        self.queue = queue
        self.wakeup_event = wakeup_event

        self.get_user_offset = get_user_offset
        self.get_io_size = get_io_size
        self.get_sequential_streak = get_sequential_streak
        self.on_error = on_error
        self._user_max_prefetch_size = user_max_prefetch_size

        self.current_offset = 0
        self.is_stopped = False
        self._active_tasks = set()
        self._producer_task = None

    @property
    def max_prefetch_size(self) -> int:
        """Calculates the maximum prefetch size based on user intent or io size.

        Returns:
            int: The maximum number of bytes to prefetch ahead.
        """
        if self._user_max_prefetch_size is not None:
            return min(
                self._user_max_prefetch_size,
                max(2 * self.get_io_size(), self.MIN_PREFETCH_SIZE),
            )
        return max(2 * self.get_io_size(), self.MIN_PREFETCH_SIZE)

    def start(self):
        """Starts the background producer loop.

        This clears any previous wakeup events and spawns the main loop task.
        """
        logger.debug("Starting PrefetchProducer loop.")
        self.is_stopped = False
        self.wakeup_event.clear()
        self._producer_task = asyncio.create_task(self._loop())

    async def stop(self):
        """Cancels all active fetch tasks and shuts down the producer loop.

        This method ensures the queue is flushed and waits for cancelled
        tasks to finish cleaning up.
        """
        logger.debug(
            "Stopping PrefetchProducer. Active fetch tasks: %d", len(self._active_tasks)
        )
        self.is_stopped = True
        self.wakeup_event.set()

        tasks_to_wait = []
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            tasks_to_wait.append(self._producer_task)

        for task in list(self._active_tasks):
            if not task.done():
                tasks_to_wait.append(task)
        self._active_tasks.clear()

        # Clear out any leftover items in the queue
        cleared_items = 0
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if (
                    isinstance(item, asyncio.Task)
                    and item.done()
                    and not item.cancelled()
                ):
                    item.exception()
                cleared_items += 1
            except asyncio.QueueEmpty:
                break

        if cleared_items > 0:
            logger.debug(
                "Cleared %d leftover items from the queue during stop.", cleared_items
            )

        if tasks_to_wait:
            logger.debug(
                "Waiting for %d cancelled tasks to finish their teardown.",
                len(tasks_to_wait),
            )
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def restart(self, new_offset: int):
        """Stops current tasks and restarts the background loop at a new byte offset.

        Args:
            new_offset (int): The new byte position to start prefetching from.
        """
        logger.debug("Restarting PrefetchProducer at new offset: %d", new_offset)
        await self.stop()
        self.current_offset = new_offset
        self.start()

    async def _loop(self):
        """The main background loop that calculates sizes and spawns fetch tasks."""
        logger.debug("PrefetchProducer internal loop is now running.")
        try:
            while not self.is_stopped:
                await self.wakeup_event.wait()
                self.wakeup_event.clear()

                if self.is_stopped:
                    break

                io_size = self.get_io_size()
                streak = self.get_sequential_streak()
                prefetch_size = min((streak + 1) * io_size, self.max_prefetch_size)

                logger.debug(
                    "Producer awake. Current offset: %d, User offset: %d, Prefetch size: %d",
                    self.current_offset,
                    self.get_user_offset(),
                    prefetch_size,
                )

                while (
                    not self.is_stopped
                    and (self.current_offset - self.get_user_offset()) < prefetch_size
                    and self.current_offset < self.size
                ):
                    user_offset = self.get_user_offset()
                    space_remaining = self.size - self.current_offset
                    prefetch_space_available = prefetch_size - (
                        self.current_offset - user_offset
                    )

                    if prefetch_size >= self.MIN_CHUNK_SIZE:
                        if prefetch_space_available >= self.MIN_CHUNK_SIZE:
                            actual_size = min(
                                max(self.MIN_CHUNK_SIZE, io_size), space_remaining
                            )
                        else:
                            break
                    else:
                        actual_size = min(io_size, space_remaining)

                    if streak < 2:
                        sfactor = self.concurrency
                    else:
                        sfactor = min(
                            self.concurrency,
                            max(
                                1,
                                actual_size * self.concurrency // prefetch_size,
                            ),
                        )

                    logger.debug(
                        "Spawning fetch task. Offset: %d, Size: %d, Split Factor: %d",
                        self.current_offset,
                        actual_size,
                        sfactor,
                    )

                    download_task = asyncio.create_task(
                        self.fetcher(
                            self.current_offset, actual_size, split_factor=sfactor
                        )
                    )
                    self._active_tasks.add(download_task)
                    download_task.add_done_callback(self._active_tasks.discard)

                    await self.queue.put(download_task)
                    self.current_offset += actual_size

        except asyncio.CancelledError:
            logger.debug("PrefetchProducer loop was cancelled.")
            pass
        except Exception as e:
            logger.error(
                "PrefetchProducer loop encountered an unexpected error: %s",
                e,
                exc_info=True,
            )
            self.is_stopped = True
            self.on_error(e)
            await self.queue.put(e)


class PrefetchConsumer:
    """Consumes prefetched chunks from the queue and manages byte slicing.

    This class pulls data out of the shared queue and slices it into the
    exact byte sizes requested by the user. It also manages the local block buffer.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        wakeup_event: asyncio.Event,
        is_producer_stopped,
        is_producer_at_eof,
        on_error,
    ):
        """Initializes the consumer.

        Args:
            queue (asyncio.Queue): The shared queue containing fetch tasks.
            wakeup_event (asyncio.Event): Event used to wake the producer when more data is needed.
            is_producer_stopped (Callable): Function returning whether the producer has been halted.
            is_producer_at_eof (Callable): Function returning whether the producer has reached the end of the file.
            on_error (Callable): Callback triggered when a fetch error is encountered.
        """
        logger.debug("Initializing PrefetchConsumer.")
        self.queue = queue
        self.wakeup_event = wakeup_event
        self.is_producer_stopped = is_producer_stopped
        self.is_producer_at_eof = is_producer_at_eof
        self.on_error = on_error
        self.sequential_streak = 0
        self.offset = 0
        self._current_block = b""
        self._current_block_idx = 0

    def seek(self, new_offset: int):
        """Clears the buffer and resets the internal offset for a hard seek.

        Args:
            new_offset (int): The byte position the consumer is jumping to.
        """
        logger.debug(
            "Consumer executing hard seek to offset %d. Clearing internal buffer.",
            new_offset,
        )
        self.offset = new_offset
        self.sequential_streak = 0
        self._current_block = b""
        self._current_block_idx = 0

    def clear_buffer(self):
        """Discards the local byte buffer. Useful during shutdown or resets."""
        logger.debug("Consumer local block buffer cleared.")
        self._current_block = b""
        self._current_block_idx = 0

    async def _advance(self, size: int, save_data: bool) -> list[bytes]:
        """Internal method to advance the offset and optionally extract data.

        Handles queue exhaustion, producer wakeups, and streak tracking.
        """
        if size <= 0:
            return []

        chunks = []
        processed = 0

        while processed < size:
            available = len(self._current_block) - self._current_block_idx

            if not available:
                if self.is_producer_stopped() and self.queue.empty():
                    logger.debug("Consumer reached EOF.")
                    break

                if self.queue.empty():
                    if self.is_producer_at_eof():
                        logger.debug("Producer is at EOF. No more data to consume.")
                        break
                    logger.debug("Queue is empty. Waking up producer.")
                    self.wakeup_event.set()

                task = await self.queue.get()

                if isinstance(task, Exception):
                    logger.error("Consumer retrieved an exception: %s", task)
                    self.on_error(task)
                    raise task

                try:
                    block = await task

                    self.sequential_streak += 1
                    if self.sequential_streak >= 2:
                        self.wakeup_event.set()

                    self._current_block = block
                    self._current_block_idx = 0
                    available = len(self._current_block)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Consumer caught an error: %s", e, exc_info=True)
                    self.on_error(e)
                    raise e

            if not self._current_block:
                break

            needed = size - processed
            take = min(needed, available)

            if save_data:
                if take == len(self._current_block) and self._current_block_idx == 0:
                    chunk = self._current_block
                else:
                    # Native Python slicing was GIL bound in my experiments.
                    chunk = await asyncio.to_thread(
                        _fast_slice, self._current_block, self._current_block_idx, take
                    )
                chunks.append(chunk)

            self._current_block_idx += take
            processed += take
            self.offset += take

        return chunks

    async def consume(self, size: int) -> bytes:
        """Pulls exactly 'size' bytes from the local block or the task queue.

        If the local block is exhausted, this will wait on the queue for the next
        available chunk of data.

        Args:
            size (int): The exact number of bytes to retrieve.

        Returns:
            bytes: The requested bytes. This may be shorter than 'size' if EOF is reached.

        Raises:
            Exception: Re-raises any exceptions encountered by the producer fetch tasks.
        """
        if size <= 0:
            return b""

        chunks = await self._advance(size, save_data=True)

        if not chunks:
            return b""

        if len(chunks) == 1:
            return chunks[0]

        return await asyncio.to_thread(b"".join, chunks)

    async def skip(self, size: int) -> None:
        """Advances the consumer offset without allocating memory."""
        await self._advance(size, save_data=False)


class BackgroundPrefetcher:
    """Orchestrator that manages reading behavior and coordinates background work.

    This acts as the main public interface for the file reader. It tracks the
    user's reading history, routes seek operations, and links the producer's
    network tasks with the consumer's data slicing logic.
    """

    def __init__(self, fetcher, size: int, concurrency: int, max_prefetch_size=None):
        """Initializes the background prefetcher.

        Args:
            fetcher (Callable): A coroutine of the form `f(start, end)` which gets bytes from the remote.
            size (int): Total byte size of the file being read.
            concurrency (int): Number of concurrent network requests to use for large chunks.
            max_prefetch_size (int, optional): Maximum bytes to prefetch ahead of the current user offset.

        Raises:
            ValueError: If max_prefetch_size is provided but is not a positive integer.
        """
        logger.debug(
            "Starting BackgroundPrefetcher. Size: %d, Concurrency: %d, Max Prefetch: %s",
            size,
            concurrency,
            max_prefetch_size,
        )
        self.size = size
        self.concurrency = concurrency

        if max_prefetch_size is not None and max_prefetch_size <= 0:
            logger.error("Invalid max_prefetch_size provided: %s", max_prefetch_size)
            raise ValueError(
                "max_prefetch_size should be a positive integer to use adaptive prefetching!"
            )

        self.loop = fsspec.asyn.get_loop()
        self._lock = threading.Lock()
        self._error = None
        self.is_stopped = False
        self.queue = asyncio.Queue()
        self.wakeup_event = asyncio.Event()
        self.user_offset = 0
        self.read_tracker = RunningAverageTracker(maxlen=10)

        self.consumer = PrefetchConsumer(
            queue=self.queue,
            wakeup_event=self.wakeup_event,
            is_producer_stopped=self._is_producer_stopped,
            is_producer_at_eof=self._is_producer_at_eof,
            on_error=self._set_error,
        )

        self.producer = PrefetchProducer(
            fetcher=fetcher,
            size=self.size,
            concurrency=self.concurrency,
            queue=self.queue,
            wakeup_event=self.wakeup_event,
            get_user_offset=lambda: self.consumer.offset,
            get_io_size=self._get_adaptive_io_size,
            get_sequential_streak=lambda: self.consumer.sequential_streak,
            on_error=self._set_error,
            user_max_prefetch_size=max_prefetch_size,
        )

        async def _start():
            self.producer.start()

        fsspec.asyn.sync(self.loop, _start)
        logger.debug("BackgroundPrefetcher initialization complete.")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point. Ensures the prefetcher is cleanly closed."""
        self.close()

    def _get_adaptive_io_size(self) -> int:
        return self.read_tracker.average

    def _is_producer_stopped(self) -> bool:
        return self.producer.is_stopped if hasattr(self, "producer") else True

    def _is_producer_at_eof(self) -> bool:
        return (
            self.producer.current_offset >= self.producer.size
            if hasattr(self, "producer")
            else True
        )

    def _set_error(self, e: Exception):
        logger.error("Global error state set in BackgroundPrefetcher: %s", e)
        self._error = e

    async def _restart_producer(self, new_offset: int):
        logger.debug(
            "Handling seek request. Restarting producer at offset: %d", new_offset
        )
        self._error = None
        await self.producer.restart(new_offset)
        self.consumer.seek(new_offset)
        self.read_tracker.clear()

    async def _async_fetch(self, start, end):
        logger.debug("Executing _async_fetch for range %d - %d.", start, end)

        if start != self.user_offset:
            if self.user_offset < start <= self.producer.current_offset:
                logger.debug(
                    "Soft seek detected. Skipping ahead from %d to %d.",
                    self.user_offset,
                    start,
                )
                skip_amount = start - self.user_offset
                await self.consumer.skip(skip_amount)
                self.user_offset = start
            else:
                logger.debug(
                    "Hard seek detected. Moving user offset from %d to %d.",
                    self.user_offset,
                    start,
                )
                self.user_offset = start
                await self._restart_producer(start)

        requested_size = end - start
        self.read_tracker.add(requested_size)

        chunk = await self.consumer.consume(requested_size)
        self.user_offset += len(chunk)

        logger.debug("Completed _async_fetch. Returned %d bytes.", len(chunk))
        return chunk

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size

        end = min(end, self.size)
        logger.debug(
            "Synchronous _fetch called for bounds start=%s, end=%s.", start, end
        )

        if start >= self.size or start >= end:
            logger.warning(
                "Invalid bounds or EOF reached in _fetch. Start: %d, End: %d, Size: %d",
                start,
                end,
                self.size,
            )
            return b""

        with self._lock:
            if self._error:
                logger.error("Cannot fetch data: instance has an active error state.")
                raise self._error

            if self.is_stopped:
                logger.error(
                    "Cannot fetch data: BackgroundPrefetcher is stopped or closed."
                )
                raise RuntimeError(
                    "The file instance has been closed. This can occur if a close operation "
                    "is executed concurrently while a read operation is still in progress."
                )

            try:
                result = fsspec.asyn.sync(self.loop, self._async_fetch, start, end)
            except Exception as e:
                logger.error(
                    "Exception raised during synchronous fetch: %s", e, exc_info=True
                )
                self.is_stopped = True
                self._error = e
                raise

            if self.is_stopped:
                logger.error("Instance was stopped mid-fetch operation.")
                raise RuntimeError(
                    "The file instance has been closed. This can occur if a close operation "
                    "is executed concurrently while a read operation is still in progress."
                )

            return result

    def close(self):
        """Safely shuts down the prefetcher.

        This cancels all background network tasks and blocks until everything
        is completely cleaned up. It also clears the internal consumer buffer.
        """
        logger.debug("Closing BackgroundPrefetcher and cleaning up resources.")
        if self.is_stopped:
            logger.debug(
                "BackgroundPrefetcher is already stopped. Skipping close operation."
            )
            return

        self.is_stopped = True
        with self._lock:
            fsspec.asyn.sync(self.loop, self.producer.stop)
            self.consumer.clear_buffer()
        logger.debug("BackgroundPrefetcher closed successfully.")
