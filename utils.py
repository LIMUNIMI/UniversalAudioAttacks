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
    Monkey-patch the nvmlDeviceGetName function in heareval.gpu_max_mem to ensure
    it always returns bytes so that the .decode() call in device_name() works correctly.
    """
    try:
        # Import the module (or get it if already imported)
        if 'heareval.gpu_max_mem' in sys.modules:
            gpu_max_mem = sys.modules['heareval.gpu_max_mem']
        else:
            import heareval.gpu_max_mem as gpu_max_mem
        
        # Check if nvmlDeviceGetName is imported in the module
        if hasattr(gpu_max_mem, 'nvmlDeviceGetName'):
            original_nvmlDeviceGetName = gpu_max_mem.nvmlDeviceGetName
            
            def patched_nvmlDeviceGetName(handle):
                result = original_nvmlDeviceGetName(handle)
                if isinstance(result, str):
                    return result.encode('utf-8')
                return result
            
            # Replace the function
            gpu_max_mem.nvmlDeviceGetName = patched_nvmlDeviceGetName
            print("Successfully patched heareval.gpu_max_mem.nvmlDeviceGetName() function")
            return True
        else:
            print("heareval.gpu_max_mem does not have nvmlDeviceGetName function")
            return False
        
    except ImportError:
        print("heareval.gpu_max_mem not available for patching (not installed yet)")
        return False
    except Exception as e:
        print(f"Error patching heareval.gpu_max_mem: {e}")
        return False


# Don't apply patch automatically - we'll apply it when needed
