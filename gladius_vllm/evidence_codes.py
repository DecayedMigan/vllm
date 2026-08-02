"""The classified refusal vocabulary for execution evidence.

Every rejection an external campaign can act on carries one of these codes as
its leading token, so a caller can distinguish "this evidence is forged" from
"this file is missing" without parsing prose. The third review required this:
a test asserting only that "an exception occurred" passes for the wrong
reason as readily as the right one.

The same strings are pinned independently in
`tests/gladius/test_third_review_adversarial.py` and in SMIG's
`smig/experience/evidence_codes.py`. Duplication is deliberate -- a rename
here must break both, rather than silently redefining the contract.
"""

from __future__ import annotations

# --- deployment expectation and receipt ----------------------------------
RECEIPT_EXPECTATION_INCOMPLETE = "RECEIPT_EXPECTATION_INCOMPLETE"
RECEIPT_IDENTITY_MISMATCH = "RECEIPT_IDENTITY_MISMATCH"
RECEIPT_DIGEST_MISMATCH = "RECEIPT_DIGEST_MISMATCH"
RECEIPT_STARTUP_MISMATCH = "RECEIPT_STARTUP_MISMATCH"
RECEIPT_PROCESS_DEAD = "RECEIPT_PROCESS_DEAD"
RECEIPT_PROCESS_REPLACED = "RECEIPT_PROCESS_REPLACED"
RECEIPT_SOCKET_OWNER_MISMATCH = "RECEIPT_SOCKET_OWNER_MISMATCH"
RECEIPT_SCHEMA_INVALID = "RECEIPT_SCHEMA_INVALID"

# --- physical GPU identity -----------------------------------------------
GPU_IDENTITY_DISAGREEMENT = "GPU_IDENTITY_DISAGREEMENT"
GPU_IDENTITY_AMBIGUOUS = "GPU_IDENTITY_AMBIGUOUS"

# --- telemetry and seals --------------------------------------------------
TELEMETRY_SCHEMA_INVALID = "TELEMETRY_SCHEMA_INVALID"
TELEMETRY_DIRECTORY_RETIRED = "TELEMETRY_DIRECTORY_RETIRED"
SEAL_SCHEMA_INVALID = "SEAL_SCHEMA_INVALID"
SEAL_SEGMENT_SET_MISMATCH = "SEAL_SEGMENT_SET_MISMATCH"
SEAL_SEGMENT_CONTENT_MISMATCH = "SEAL_SEGMENT_CONTENT_MISMATCH"
SEAL_BOUNDS_MISMATCH = "SEAL_BOUNDS_MISMATCH"
SEAL_INSTANCE_MISMATCH = "SEAL_INSTANCE_MISMATCH"
SEAL_RECEIPT_EXPECTATION_MISMATCH = "SEAL_RECEIPT_EXPECTATION_MISMATCH"
SEAL_APPLICATION_SEMANTIC_MISMATCH = "SEAL_APPLICATION_SEMANTIC_MISMATCH"
SEAL_APPLICATION_STEP_UNSEALED = "SEAL_APPLICATION_STEP_UNSEALED"

# --- application acknowledgement ------------------------------------------
APPLICATION_SCHEMA_INVALID = "APPLICATION_SCHEMA_INVALID"

# --- seal request lifecycle ------------------------------------------------
SEAL_REQUEST_SCHEMA_INVALID = "SEAL_REQUEST_SCHEMA_INVALID"
SEAL_REQUEST_FOREIGN_INSTANCE = "SEAL_REQUEST_FOREIGN_INSTANCE"
SEAL_REQUEST_STALE_GENERATION = "SEAL_REQUEST_STALE_GENERATION"
SEAL_REQUEST_DEPLOYMENT_CHANGED = "SEAL_REQUEST_DEPLOYMENT_CHANGED"
SEAL_REQUEST_AFTER_RETIREMENT = "SEAL_REQUEST_AFTER_RETIREMENT"


def classify(code: str, message: str) -> str:
    """One error string in the form the whole protocol agrees on."""
    return f"{code}: {message}"
