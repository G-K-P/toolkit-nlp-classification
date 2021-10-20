import time
from utils.console import console


def loading(func):
    def inner(*args, **kwargs):
        start = time.perf_counter()
        status = "Loading" if "task_name" not in kwargs else kwargs["task_name"]
        with console.status(status) as _:
            result = func(*args, **kwargs)
        end = time.perf_counter()
        console.log(f"Finished in {round(end - start, 2)} seconds")
        return result

    return inner
