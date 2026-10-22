"""Marks the test suite as a package so `tests.<module>` names this repo's helpers.

Without it `tests` is a namespace package, and the editable install of the sibling
`pydantic-ai-app-factory` -- whose repo root lands on `sys.path` and whose own `tests` *is* a
package -- wins the import, so `tests.containers` and `tests.sql_stack` resolve to nothing.
"""
