"""MemRec memory module.

The package used to import the torch-dependent pruner eagerly.  Keeping these
exports lazy lets offline artifact commands reuse ``SnippetPacker`` without
requiring a training environment, while preserving the public imports used by
the live MemRec pipeline.
"""

__all__ = [
    "UserItemGraph",
    "NeighborPruner",
    "SnippetPacker",
    "MemRecManager",
    "MemoryStorage",
    "FacetEncoder",
    "TextBundleBuilder",
]


def __getattr__(name):
    if name == "UserItemGraph":
        from .graph import UserItemGraph
        return UserItemGraph
    if name == "NeighborPruner":
        from .pruner import NeighborPruner
        return NeighborPruner
    if name == "SnippetPacker":
        from .packer import SnippetPacker
        return SnippetPacker
    if name == "MemRecManager":
        from .manager import MemRecManager
        return MemRecManager
    if name == "MemoryStorage":
        from .storage import MemoryStorage
        return MemoryStorage
    if name == "FacetEncoder":
        from .encoder import FacetEncoder, TextBundleBuilder
        return {"FacetEncoder": FacetEncoder, "TextBundleBuilder": TextBundleBuilder}[name]
    if name == "TextBundleBuilder":
        from .encoder import FacetEncoder, TextBundleBuilder
        return {"FacetEncoder": FacetEncoder, "TextBundleBuilder": TextBundleBuilder}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
