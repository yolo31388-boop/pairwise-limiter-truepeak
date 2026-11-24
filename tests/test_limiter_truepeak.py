import pytest
import media.limiter_truepeak as mod

CASES = [
    ("true_peak_detect", (1,)),
    ("lookahead_samples", (1,)),
    ("gain_cross_block", (1,)),
    ("multichannel_link", (1,)),
    ("limit_after_resample", (1,)),
    ("smooth_release", (1,)),
    ("process", (1,)),
    ("compute_gain", (1,)),
    ("apply", (1,)),
    ("export", (1,)),
    ("validate_limit", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
