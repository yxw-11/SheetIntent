"""Legacy import path for code extraction; execution lives in the harness."""
import re

def extract_code(response: str) -> str:
    """Extract executable Python from model output.

    Models sometimes omit the closing markdown fence when they hit max_tokens.
    In that case, keep the text after the last opening fence instead of trying
    to execute the literal ```python marker.
    """

    response = response.strip()
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()

    opening = list(re.finditer(r"```(?:python|py)?\s*", response, flags=re.IGNORECASE))
    if opening:
        return response[opening[-1].end() :].strip().rstrip("`").strip()

    code_start = _first_code_line(response)
    if code_start >= 0:
        return response[code_start:].strip()
    return response


def _first_code_line(text: str) -> int:
    for match in re.finditer(r"(?m)^(from\s+\S+\s+import\s+|import\s+\S+|def\s+solve\s*\()", text):
        return match.start()
    return -1
