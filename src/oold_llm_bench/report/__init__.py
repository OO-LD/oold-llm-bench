"""Reading results back as a table.

Rows are keyed on the whole condition. A mean pooled across models or
catalogue sizes can move for reasons nobody can name.
"""

from oold_llm_bench.report.table import Cell, Table, tabulate

__all__ = ["Cell", "Table", "tabulate"]
