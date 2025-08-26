import sys
import subprocess


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()  # Ensure immediate output

    def flush(self):
        for f in self.files:
            f.flush()


def run_command(cmd):
    print(f"Running command: {cmd}")  # Log the command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: {cmd}"
        )
    return result.returncode
