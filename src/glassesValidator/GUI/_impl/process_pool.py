import pebble
import multiprocessing
import typing
import threading


if __name__ in ["__main__","__mp_main__"]:
    # to allow running this example code directly
    import sys
    import pathlib

    self_path = pathlib.Path(__file__).parents[3]
    sys.path.append(str(self_path))

    from glassesValidator.GUI._impl.structs import CounterContext, ProcessState
    from glassesValidator.GUI._impl import globals
else:
    from .structs import CounterContext, ProcessState
    from . import globals


done_callback: typing.Callable = None

# NB: pool is only started in run() once needed
_pool: pebble.pool.process.ProcessPool = None
_work_items: dict[int,pebble.ProcessFuture] = None
_work_id_provider: CounterContext = CounterContext()
_lock: threading.Lock = threading.Lock()

def _cleanup():
    global _pool
    global _work_items

    # cancel all pending and running jobs
    cancel_all_jobs()

    # stop pool
    if _pool and _pool.active:
        _pool.stop()
        _pool.join()
    _pool = None
    _work_items = None

def cleanup():
    global _lock

    with _lock:
        _cleanup()

def cleanup_if_no_work():
    global _lock
    with _lock:
        if _pool and not _work_items:
            _cleanup()

class ProcessWaiter(object):
    """Routes completion through to user callback."""
    def add_result(self, future):
        self._notify(future, ProcessState.Completed)

    def add_exception(self, future):
        self._notify(future, ProcessState.Failed)

    def add_cancelled(self, future):
        self._notify(future, ProcessState.Canceled)

    def _notify(self, future, state: ProcessState):
        global _lock

        with _lock:
            id = None
            if _work_items is not None and future in _work_items.values():
                id = list(_work_items.keys())[list(_work_items.values()).index(future)]

            if id is None:
                return

            # clean up the work item since we're done with it
            del _work_items[id]

            # execute user callback, if any
            if done_callback:
                done_callback(future, id, state)

def run(fn: typing.Callable, *args, **kwargs):
    global _pool
    global _work_items
    global _work_id_provider
    global _lock

    with _lock:
        if _pool is None or not _pool.active:
            context = multiprocessing.get_context("spawn")  # ensure consistent behavior on Windows (where this is default) and Unix (where fork is default, but that may bring complications)
            if globals.settings is None:
                max_workers = 2
            else:
                max_workers = globals.settings.process_workers
            _pool = pebble.ProcessPool(max_workers=max_workers, context=context)

        if _work_items is None:
            _work_items = {}

        with _work_id_provider:
            work_id = _work_id_provider.get_count()
            _work_items[work_id] = _pool.submit(fn, None, *args, **kwargs)
            _work_items[work_id]._waiters.append(ProcessWaiter())
            return work_id

def _get_status_from_future(fut: pebble.ProcessFuture):
    if fut.running():
        return ProcessState.Running
    elif fut.done():
        if fut.cancelled():
            return ProcessState.Canceled
        elif fut.exception() is not None:
            return ProcessState.Failed
        else:
            return ProcessState.Completed
    else:
        return ProcessState.Pending


def get_job_state(id: int):
    if not _work_items:
        return None
    fut = _work_items.get(id, None)
    if fut is None:
        return None
    else:
        return _get_status_from_future(fut)

def cancel_job(id: int):
    if not _work_items:
        return False
    if (future := _work_items.get(id, None)) is None:
        return False

    return future.cancel()

def cancel_all_jobs():
    if not _work_items:
        return
    for id in reversed(_work_items):    # reversed so that later pending jobs don't start executing when earlier gets cancelled, only to be canceled directly after
        if not _work_items[id].done():
            _work_items[id].cancel()




# Example usage
def _do_work(id,seconds):   # this needs to be at module level so it can be run in a separate process
    import time
    print(f"{id}: Going to sleep {seconds}s..")
    if id==3:
        raise RuntimeError('boom')
    time.sleep(seconds)
    print(f"{id}: Slept {seconds}s.")


if __name__ == "__main__":
    import asyncio
    from glassesValidator.GUI._impl import async_thread, process_pool, utils

    async_thread.setup()

    # example user callback
    def worker_process_done_hook(future: pebble.ProcessFuture, id: int, state: ProcessState):
        match state:
            case ProcessState.Canceled:
                print(f'{id}: {state.name}')
            case ProcessState.Completed:
                print(f'{id}: {state.name}')
            case ProcessState.Failed:
                print(f'{id}: {state.name}')
                exc = future.exception()    # should not throw exception since CancelledError is already encoded in state and future is done
                tb = utils.get_traceback(type(exc), exc, exc.__traceback__)
                print(f"Something went wrong in a worker process for work item {id}:\n\n{tb}")
    process_pool.done_callback = worker_process_done_hook

    async def my_main():
        def print_job_states(work_ids):
            for id in work_ids:
                if (state:=process_pool.get_job_state(id)) is not None:
                    print(f'{id}: {state.name}')
            print('-----')

        work_ids = []
        # 1. enqueue some work
        for id in range(5):
            # NB: you should keep your own task to work_id mapping if you need it. I'm skipping over that here
            work_ids.append(process_pool.run(_do_work,id,3))

        # wait long enough for some of the work to finish and see what states the tasks have
        await asyncio.sleep(5)
        print_job_states(work_ids)

        # 2. cancel whatever is outstanding, and see what states we have then
        print('cancelling jobs')
        for id in reversed(work_ids):
            process_pool.cancel_job(id)

        # 3. enqueue some more work, see what states we have
        work_ids = []   # NB: even though we dump the work_ids from the previous jobs here, there state is still kept in process_pool internally. Use process_pool.clear_job_state() to clear that out if you really think its needed
        print('enqueueing some more jobs')
        for id in range(5,9):
            work_ids.append(process_pool.run(_do_work,id,3))
        # wait long enough for some work to start but not finish and see what states the tasks have
        await asyncio.sleep(2)
        print_job_states(work_ids)

        # 4. cancel everything, and see what states we have then
        print('cancelling all jobs')
        process_pool.cancel_all_jobs()

        # 5. make sure you clean up before exiting, else you may get some nasty crashes as process
        # scaffolding dissappears before all tasks are done
        print('exiting...')
        process_pool.cleanup()

    # run in async so that time.sleep() calls can be avoided, since doing these in main would block execution of _check_status_update coroutine in the async thread
    async_thread.wait(my_main())
