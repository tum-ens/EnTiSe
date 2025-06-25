"""EnTiSe - Energy Time Series Generator."""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("entise")
    except PackageNotFoundError:
        __version__ = "unknown"
except ImportError:
    __version__ = "unknown"

# Import and expose the methods module
import entise.methods
