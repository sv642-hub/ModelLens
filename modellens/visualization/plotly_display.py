"""
Display Plotly figures in Jupyter without Plotly's MIME path (no ``nbformat``).

Kept separate from ``common.py`` so imports never depend on heavy helpers or
package init order.
"""

from __future__ import annotations

from typing import Any


def showfig(fig: Any) -> None:
    try:
        from IPython.display import HTML, display
    except ImportError:
        fig.show()
        return
    display(HTML(fig.to_html(include_plotlyjs="cdn")))
