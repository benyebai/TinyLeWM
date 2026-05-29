import numpy as np

# Which bit equals which action
# [Left Right Up Down A B]
BIT_LOOKUP = np.array([5, 2, 6, 1, 7, 4], dtype=np.uint8)


# takes a 20 -> maps to an action based on its bit version
# returns: a numpy array (cool thing is, its actually stored in one continous block of memory)
def action_code_to_multihot(code: int) -> np.ndarray:
    # turn it into bit wise, look at the bit at that index
    # ex. 20 -> 10100 -> for i in BIT_LOOKUP: delete that many bits from the end
    # (& 1) take the last bit
    return ((int(code) >> BIT_LOOKUP) & 1).astype(np.uint8)


def action_codes_to_multihot(codes: np.ndarray) -> np.ndarray:
    # asarray: same buffer, dosent copy | array: will always coppy
    codes = np.asarray(codes, dtype=np.int64)
    # [N rows, 1 col] >> [1 row, 6 col] -> [N rows, 6 cols]
    return ((codes[:, None] >> BIT_LOOKUP[None, :]) & 1).astype(np.uint8)
