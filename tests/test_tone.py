"""The tone map: picture -> spacing r(x)."""

import numpy as np

from unbroken_line import Growth, flatness, load_catalog, params, spacing_field


def ramp(h=60, w=80):
    return np.tile(np.linspace(0.0, 1.0, w), (h, 1))


def test_params_order_defaults_then_artwork_then_knobs():
    entry = {"growth": {"gamma": 3.0, "local": 1.5}}
    g = params(entry, local=0.2)
    assert g.gamma == 3.0 and g.local == 0.2 and g.r_min == Growth().r_min
    assert params() == Growth()


def test_catalog_overrides_are_growth_fields():
    fields = set(Growth.__dataclass_fields__)
    for slug, entry in load_catalog().items():
        assert set(entry.get("growth", {})) <= fields, slug
        params(entry)


def test_spacing_spans_r_min_to_r_max_and_follows_tone():
    g = Growth(width=80, local=0.0, blur=0.0)
    R = spacing_field(ramp(), g)
    assert R.min() == g.r_min and R.max() == g.r_max
    assert np.all(np.diff(R[30]) >= -1e-9)  # darker left, tighter strands


def test_gamma_keeps_more_dense():
    a = spacing_field(ramp(), Growth(width=80, local=0.0, gamma=1.0))
    b = spacing_field(ramp(), Growth(width=80, local=0.0, gamma=2.0))
    assert b.mean() < a.mean()


def test_flatten_lifts_a_plain_ground_but_not_a_textured_subject():
    rng = np.random.default_rng(0)
    img = np.full((80, 80), 0.2)
    img[:, 40:] += 0.15 * rng.standard_normal((80, 40))  # textured right half
    g = Growth(width=80, r_max=8.0, local=0.0, flatten=0.9, flatten_size=1.0)
    f = flatness(img, g)
    assert f[:, 5:30].mean() > 0.8 and f[:, 50:75].mean() < 0.3
    flat = spacing_field(img, Growth(width=80, r_max=8.0, local=0.0))
    lifted = spacing_field(img, g)
    assert lifted[:, 5:30].mean() > flat[:, 5:30].mean() + 2.0  # plain ground opens up


def test_local_contrast_keeps_a_dark_spot_in_a_light_field():
    img = np.full((80, 80), 0.8)
    img[:, :20] = 0.0  # a dark band, so the stretch has a black to anchor
    img[38:42, 58:62] = 0.5  # an "eye": mid-grey in a lit "cheek"
    flat = spacing_field(img, Growth(width=80, local=0.0, gamma=1.0))
    loc = spacing_field(img, Growth(width=80, local=1.5, gamma=1.0))
    gap = lambda R: R[20, 60] - R[40, 60]  # cheek minus eye
    assert gap(loc) > gap(flat)
