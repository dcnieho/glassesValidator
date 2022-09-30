import pebble
import multiprocessing
import aioprocessing
import asyncio
import typing
import concurrent.futures


if __name__ in ["__main__","__mp_main__"]:
    # to allow running this example code directly
    import sys
    import pathlib

    self_path = pathlib.Path(__file__).parents[3]
    sys.path.append(str(self_path))
    
    from glassesValidator.GUI._impl.structs import CounterContext, ProcessState
    from glassesValidator.GUI._impl import async_thread, globals
else:
    from .structs import CounterContext, ProcessState
    from . import async_thread, globals


done_callback: typing.Callable = None

_pool: pebble.pool.process.ProcessPool = None
_work_items: dict[int,pebble.ProcessFuture] = None
_work_id_provider: CounterContext = None
_status_dict: dict[int,ProcessState] = None
_status_queue: aioprocessing.managers.AioQueueProxy = None
_status_update_coro: concurrent.futures.Future = None


def setup():
    global _work_items
    global _work_id_provider
    global _status_dict
    global _status_queue

    _work_items = {}
    _work_id_provider = CounterContext()

    _status_dict = {}

    # NB: pool and status checker are only started in run() once needed

def cleanup():
    global _pool
    global _work_items
    global _work_id_provider
    global _status_dict
    global _status_queue
    global _status_update_coro

    # cancel all pending and running jobs
    cancel_all_jobs()

    # stop pool
    if _pool and _pool.active:
        _pool.stop()
        _pool.join()
    _pool = None
    _status_dict = None
    _work_items = None
    _work_id_provider = None

    # tell status queue checker to shut down
    if _status_update_coro and not _status_update_coro.done():
        _status_queue.put(None)
    _status_update_coro = None
    _status_queue = None

def _work_bootstrapper(status_queue,id, fn,args,kwargs):
    status_queue.put({id: ProcessState.Running})
    fn(*args,**kwargs)

def _process_done_hook(future: pebble.ProcessFuture):
    global _status_queue
    id = None
    if _work_items is not None and future in _work_items.values():
        id = list(_work_items.keys())[list(_work_items.values()).index(future)]

    try:
        exc = future.exception()
    except asyncio.CancelledError:
        state = ProcessState.Canceled
    else:
        if exc:
            state = ProcessState.Failed
        else:
            state = ProcessState.Completed
            
    # NB: even though this hook runs in the parent process and could thus directly change the state in status_dict,
    # I still use the status_queue. This way there is only one point where there status_dict is changed, keeping
    # program flow easier to understand
    if id is not None and _status_queue is not None:
        _status_queue.put({id: state})

    # execute user callback, if any
    if done_callback:
        done_callback(future, id, state)

async def _check_status_update(status_dict, status_queue):
    while (item := await status_queue.coro_get()) is not None:
        for id in item:
            if id not in status_dict or status_dict[id].value<item[id].value:   # make sure we don't go backward in state
                status_dict[id] = item[id]

                # execute user callback, if any, to notify state change
                # but only for pending and running state, completed, canceled and failed are notified by _process_done_hook
                if done_callback:
                    done_callback(_work_items[id], id, item[id])

def run(fn: typing.Callable, *args, **kwargs):
    global _pool
    global _work_items
    global _work_id_provider
    global _status_update_coro
    global _status_dict
    global _status_queue
    global _status_update_coro

    if _status_queue is None:
        _status_queue = aioprocessing.AioManager().AioQueue()
    if _pool is None or not _pool.active:
        context = multiprocessing.get_context("spawn")  # ensure consistent behavior on Windows (where this is default) and Unix (where fork is default, but that may bring complications)
        if globals.settings is None:
            max_workers = 2
        else:
            max_workers = globals.settings.process_workers
        _pool = pebble.ProcessPool(max_workers=max_workers, context=context)

    if _status_update_coro is None or _status_update_coro.done():
        _status_update_coro = async_thread.run(_check_status_update(_status_dict, _status_queue))
        
    with _work_id_provider:
        work_id = _work_id_provider.get_count()
        # route function execution through _work_bootstrapper() so that we get a notification that the work item is started to be processed once a worker takes it up
        _work_items[work_id] = async_thread.loop.run_in_executor(_pool, _work_bootstrapper, None, _status_queue, work_id, fn, args, kwargs)
        _status_queue.put({work_id: ProcessState.Pending})
        _work_items[work_id].add_done_callback(_process_done_hook)
        return work_id

def get_job_state(id: int):
    return _status_dict.get(id, None)

def clear_job_state(id: int):
    if id in _status_dict:
        del _status_dict[id]

def cancel_job(id: int):
    if (future := _work_items.get(id, None)) is None:
        return False
    
    return future.cancel()

def cancel_all_jobs():
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
    from glassesValidator.GUI._impl import process_pool, utils

    async_thread.setup()
    process_pool.setup()

    # example user callback
    def worker_process_hook(future: pebble.ProcessFuture, id: int, state: ProcessState):
        if state==ProcessState.Failed:
            exc = future.exception()    # should not throw exception since CancelledError is already encoded in state and future is done
            tb = utils.get_traceback(type(exc), exc, exc.__traceback__)
            print(f"Something went wrong in a worker process for work item {id}:\n\n{tb}")
    process_pool.done_callback = worker_process_hook

    async def my_main():
        def print_job_states(work_ids):
            for id in work_ids:
                print(f'{id}: {process_pool.get_job_state(id).name}')
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
        await asyncio.sleep(.2)  # little bit of time to make jobs have been canceled before we check their state
        print_job_states(work_ids)
    
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
        await asyncio.sleep(.2)  # little bit of time to make jobs have been canceled before we check their state
        print_job_states(work_ids)

        # 5. make sure you clean up before exiting, else you'll get some nasty crashes as process
        # scaffolding dissappears before all tasks are done
        print('exiting...')
        process_pool.cleanup()

    # run in async so that time.sleep() calls can be avoided, since doing these in main would block execution of _check_status_update coroutine in the async thread
    async_thread.wait(my_main())
