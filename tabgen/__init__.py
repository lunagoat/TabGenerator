"""TabGenerator: audio in, tabs out."""
from . import polite

polite.limit_cpu()  # before numpy/torch start their thread pools
