"""Shared transport and result-size limits.

These constants live in the lightweight core package so the contributor runtime
and coordinator enforce the same signed-envelope boundary without importing
server-only dependencies.
"""

MAX_RESULT_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_SIGNED_ENVELOPE_HEADROOM_BYTES = 256 * 1024
MAX_HTTP_BODY_BYTES = MAX_RESULT_OUTPUT_BYTES + MAX_SIGNED_ENVELOPE_HEADROOM_BYTES
MAX_CONTROL_REQUEST_BYTES = MAX_HTTP_BODY_BYTES
