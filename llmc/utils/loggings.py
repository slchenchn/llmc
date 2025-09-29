import warnings
from pathlib import Path

from loguru import logger

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


class WandbLogger:
    """
    A class to handle wandb logging with error handling and graceful fallbacks.
    """

    def __init__(self):
        self._wandb_available = False
        self._wandb_initialized = False
        self._wandb = None

        try:
            import wandb

            self._wandb = wandb
            self._wandb_available = True
        except ImportError:
            self._wandb = None

    def setup(self, args, config):
        """
        Setup wandb with error handling.

        Args:
            args: Command line arguments containing config path
            config: Configuration dictionary

        Returns:
            bool: True if wandb setup succeeded, False otherwise
        """
        if not self._wandb_available:
            logger.warning("wandb is not available. Logging will be skipped.")
            return False

        try:
            self._wandb.login()
            cfg_path = Path(args.config)

            self._wandb.init(
                project="llmc",
                group=cfg_path.parent.name,
                name=cfg_path.stem,
                config=config,
            )
            self._wandb_initialized = True
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Logging will be skipped.")
            self._wandb_initialized = False
            return False

    def log(self, data, step=None):
        """
        Safely log data to wandb with error handling.

        Args:
            data: Dictionary of data to log
            step: Optional step number
        """
        if not self._wandb_available or not self._wandb_initialized:
            return

        try:
            if step is not None:
                self._wandb.log(data, step=step)
            else:
                self._wandb.log(data)
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    def is_available(self):
        """
        Check if wandb is available and initialized.

        Returns:
            bool: True if wandb is available and initialized
        """
        return self._wandb_available and self._wandb_initialized

    def finish(self):
        """
        Finish the wandb run.
        """
        if self._wandb_available and self._wandb_initialized:
            try:
                self._wandb.finish()
                self._wandb_initialized = False
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")


# Global wandb logger instance
wandb_logger = WandbLogger()


# Backward compatibility functions
def setup_wandb(args, config):
    """
    Setup wandb with error handling.

    Args:
        args: Command line arguments containing config path
        config: Configuration dictionary

    Returns:
        bool: True if wandb setup succeeded, False otherwise
    """
    return wandb_logger.setup(args, config)


def safe_wandb_log(data, step=None):
    """
    Safely log data to wandb with error handling.

    Args:
        data: Dictionary of data to log
        step: Optional step number
    """
    wandb_logger.log(data, step)


def is_wandb_available():
    """
    Check if wandb is available and initialized.

    Returns:
        bool: True if wandb is available and initialized
    """
    return wandb_logger.is_available()
