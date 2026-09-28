"""Number formatting shared by user-visible backend strings."""


def indian_grouping(value: int) -> str:
    """Group digits the Indian way (2,23,676), matching the frontend's en-IN.

    Python's `{:,}` gives 223,676, so a reason citing a rating read differently
    from the review count printed beside it on the same card.
    """
    sign = "-" if value < 0 else ""
    digits = str(abs(int(value)))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    pairs = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    return sign + ",".join([head, *pairs, tail])
