REGISTRY = {}

def register(func):
    """
    A decorator factory used to register a function.

    The decorated function is stored in the global REGISTRY dictionary
    under the provided function's __name__ attribute..

    Returns:
        Callable: A decorator function that takes a function and registers it.
    """
    # print("registering", func.__name__)
    REGISTRY[func.__name__] = func
    return func