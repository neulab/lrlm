"""
Math Utilities
"""

__all__ = ['ceil_div']


def ceil_div(a, b):
    """
    Integer division that rounds up.
    """
    return (a + b - 1) // b
