# 0005. Return detections as JSON, not a drawn image

Status: accepted (2026-10)

## Context

The 2024 API returned the input image with boxes drawn on it, base64-encoded inside a CSV
response. That works for a demo page and nothing else. A client that wants to count people,
raise an alert for a forklift, or store results for an audit has to run its own detector on the
picture we sent back.

## Decision

The model server returns, per image, `width`, `height` and a list of
`{label, label_id, score, box: [x_min, y_min, x_max, y_max]}` in pixels. The confidence
threshold is a request parameter (default 0.7). Drawing happens in the client; the Streamlit UI
is one such client.

## Consequences

- Responses are a few hundred bytes instead of a re-encoded image, and downstream systems can
  use them directly.
- The threshold is a business setting per caller: a safety alert wants recall (low threshold),
  a count on a dashboard wants precision (high threshold). The same model serves both.
- Breaking change for the old request format. The server still accepts base64 with line breaks
  (what the old client sent), but the response shape is new.
