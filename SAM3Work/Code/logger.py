"""Logger for training progress and metrics."""

import json
import os
import time


class Logger:
    """Logs training metrics to both text file and JSONL format."""

    def __init__(self, out):
        """Initialize logger with output directory.
        
        Args:
            out: Output directory path where log files will be created
        """
        self.f = open(os.path.join(out, "train_log.txt"), "a")
        self.j = open(os.path.join(out, "train_log.jsonl"), "a")

    def __call__(self, msg):
        """Log a message with timestamp.
        
        Args:
            msg: Message to log
        """
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def record(self, **row):
        """Record structured metrics as JSON.
        
        Args:
            **row: Key-value pairs to record
        """
        self.j.write(json.dumps(row) + "\n")
        self.j.flush()
