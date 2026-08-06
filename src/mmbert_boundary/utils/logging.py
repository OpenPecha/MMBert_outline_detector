"""Training log tee: duplicates stdout/stderr to a persistent log file."""

import sys
import time
from pathlib import Path


class _TeeStream:
    """Duplicates writes to both the original stream and a log file.

    Every write is flushed immediately so the log file stays current
    even if the process is killed mid-training.
    """

    def __init__(self, original_stream, log_file) -> None:
        self._original = original_stream
        self._log_file = log_file

    def write(self, msg: str) -> int:
        """Write to both the original stream and the log file.

        Args:
            msg: Text to write.

        Returns:
            Number of characters written.
        """
        self._original.write(msg)
        self._original.flush()
        self._log_file.write(msg)
        self._log_file.flush()
        return len(msg)

    def flush(self) -> None:
        """Flush both underlying streams."""
        self._original.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        """Return the file descriptor of the original stream."""
        return self._original.fileno()

    def isatty(self) -> bool:
        """Return whether the original stream is a TTY."""
        return self._original.isatty()


def setup_logging(output_dir: Path) -> None:
    """Tee stdout and stderr to ``training.log`` under *output_dir*.

    All print / tqdm / warning output is captured in real time.

    Args:
        output_dir: Directory where ``training.log`` will be created or
            appended to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    log_fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    log_fh.write(f"\n{'=' * 70}\n")
    log_fh.write(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fh.write(f"{'=' * 70}\n")
    log_fh.flush()

    sys.stdout = _TeeStream(sys.__stdout__, log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, log_fh)

    print(f"Logging to {log_path}")
