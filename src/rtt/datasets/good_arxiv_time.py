"""GOOD-Arxiv time/concept profile."""
from .good_arxiv import load as _load


def load(root, options=None, download=False):
    return _load(root, options, download, domain="time")
