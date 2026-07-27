"""Regression tests for the MS mass-agreement tolerance in the cid bridge.

The bridge previously accepted a mass match within a flat ±5 Da. Measured
against BindingDB, 19.9% of Google-derived records were right chemistry
attached to the wrong compound ID — a window that wide is the mechanism.
These tests pin the tolerance to the reported measurement regime so nobody
re-widens it without a failing test.
"""
from patentdb.routes.process_patent import ms_tolerance_da


def test_integer_mz_gets_half_dalton():
    """Unit-resolution LC-MS, the common case in synthesis examples."""
    assert ms_tolerance_da(523.0) == 0.5
    assert ms_tolerance_da(412.0) == 0.5


def test_one_or_two_decimals_gets_nominal_window():
    assert ms_tolerance_da(523.2) == 0.3
    assert ms_tolerance_da(523.25) == 0.3


def test_hrms_gets_ppm_window():
    """Four decimal places means HRMS; 20 ppm is the conventional band."""
    tol = ms_tolerance_da(523.2451)
    assert tol == 523.2451 * 20e-6
    # ~0.010 Da at this mass — three orders tighter than the old ±5 Da.
    assert 0.005 < tol < 0.02


def test_every_regime_is_far_tighter_than_the_old_flat_window():
    """The whole point: nothing may return anything close to ±5 Da."""
    for value in (150.0, 523.0, 523.2, 523.2451, 899.0, 1200.4567):
        assert ms_tolerance_da(value) < 1.0


def test_old_window_would_have_accepted_a_wrong_analog():
    """A +4 Da difference is a different molecule, not a measurement error.

    Two analogs separated by 4 Da (e.g. a ring saturation difference) both
    passed the old flat gate. Under the regime-aware tolerance neither does.
    """
    stored_mw = 522.0
    wrong_analog_mh = 526.0          # (526.0 - 1.008) - 522.0 = +2.99 Da off
    delta = abs((wrong_analog_mh - 1.008) - stored_mw)
    assert delta <= 5.0                              # old gate: accepted
    assert delta > ms_tolerance_da(wrong_analog_mh)  # new gate: rejected


def test_a_genuine_match_still_passes():
    """Tightening must not reject real agreement."""
    stored_mw = 522.2400
    reported_mh = 523.2478           # protonated, within a few ppm
    delta = abs((reported_mh - 1.008) - stored_mw)
    assert delta <= ms_tolerance_da(reported_mh)


def test_degenerate_input_does_not_crash():
    assert ms_tolerance_da(0.0) == 0.5
    assert ms_tolerance_da(-1.0) == 0.5
