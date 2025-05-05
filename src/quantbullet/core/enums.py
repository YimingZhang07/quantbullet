from enum import Enum

class ParallelizationMode(Enum):
    """Enum for parallelization modes."""
    AUTO = "auto"  # Automatically determine the best mode based on the environment
    MULTI_PROCESS = "multi_process"  # Use multiple processes for parallelization
    MULTI_THREAD = "multi_thread"  # Use multiple threads for parallelization
    SINGLE_THREAD = "single_thread"  # Use a single thread for execution