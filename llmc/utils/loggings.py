import warnings

_warning_messages = set()


def warning_once(message: str):
    """
    Show a warning message once.

    Args:
        message (str): The warning message to be shown.
    """
    if message not in _warning_messages:
        warnings.warn(message)
        _warning_messages.add(message) 