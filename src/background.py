import asyncio
import threading
import streamlit as st

_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

def submit(coro):
    task_holder = {}
    
    async def wrapper():
        task = asyncio.current_task()
        task_holder["task"] = task
        return await coro
    
    future = asyncio.run_coroutine_threadsafe(wrapper(), _loop)
    future.task_holder = task_holder  # attach it to the future
    return future

def cancel(future):
    task = future.task_holder.get("task")
    if task:
        _loop.call_soon_threadsafe(task.cancel)
    else:
        # task hasn't started yet, cancel the future directly
        future.cancel()

def get_state(*keys: str, default=None):
    d = st.session_state[st.session_state.selected_project['id']]
    for key in keys[:-1]:
        d = d.get(key, dict())
    return d.get(keys[-1], default)

def set_state(*keys: str, value, aggressive=True):
    d = st.session_state[st.session_state.selected_project['id']]
    if aggressive:
        for key in keys[:-1]:
            d = d.setdefault(key, dict())
        d[keys[-1]] = value

def delete_value(*keys):
    d = st.session_state[st.session_state.selected_project['id']]
    for key in keys[:-1]:
        d = d[key]
    del d[keys[-1]]

def close_dialog():
    set_state('active_dialog', value=None)
    st.rerun(scope='app')

def submit_coroutines():
    to_submit = get_state('coro_queue')
    while not to_submit.empty():
        type, coro = to_submit.get_nowait()
        get_state('running_coroutines')[type].append(submit(coro))

def decorated_wrapper(decorator, func):
    @decorator
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if 'streamlit' in type(e).__module__.lower():
                raise  # let all Streamlit internal exceptions through
            return
    return wrapper

def fragment(fragment_func=None, *, run_every: float = 1):
    if fragment_func is None:
        def new_decorator(func):
            return decorated_wrapper(st.fragment(run_every=run_every), func)
        return new_decorator
    else:
        return decorated_wrapper(st.fragment, fragment_func)