import sys
import subprocess
import importlib


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


# Patch the heareval.gpu_max_mem module dynamically
def patch_gpu_max_mem_dynamically():
    """
    Monkey-patch the device_name method in heareval.gpu_max_mem to handle
    both bytes and string returns from nvmlDeviceGetName.
    """
    try:
        # Import the module (or get it if already imported)
        if 'heareval.gpu_max_mem' in sys.modules:
            gpu_max_mem = sys.modules['heareval.gpu_max_mem']
        else:
            # Import the module to patch it
            import heareval.gpu_max_mem as gpu_max_mem
        
        # Store the original method
        original_device_name = gpu_max_mem.GPUMaxMem.device_name
        
        # Create a patched version
        def patched_device_name(self):
            name = original_device_name(self)
            if isinstance(name, bytes):
                return name.decode("utf-8")
            else:
                return name  # already a string
        
        # Replace the method
        gpu_max_mem.GPUMaxMem.device_name = patched_device_name
        print("Successfully patched heareval.gpu_max_mem.device_name() dynamically")
        return True
        
    except ImportError:
        print("heareval.gpu_max_mem not available for patching (not installed yet)")
        return False
    except Exception as e:
        print(f"Error patching heareval.gpu_max_mem: {e}")
        return False


# Apply the patch when utils is imported
patch_gpu_max_mem_dynamically()
