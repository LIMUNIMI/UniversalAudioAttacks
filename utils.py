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
    """Run a shell command and print its output to both terminal and log"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode
