from types import SimpleNamespace

from cvti.serving.concealment_notices import ConcealmentNotices


def alert(camera="one", detector="concealment"):
    return SimpleNamespace(camera_id=camera, payload={
        "candidate": SimpleNamespace(detector=detector)})


def test_unverified_candidate_shows_nothing():
    notices = ConcealmentNotices()
    notices.candidate(alert())
    assert notices.snapshot("one") is None


def test_confirmed_notice_expires_and_does_not_cross_cameras():
    now = [100]
    notices = ConcealmentNotices(lambda: now[0])
    a = alert()
    notices.candidate(a)
    notices.verdict(a, SimpleNamespace(confirmed=True, errored=False))
    assert notices.snapshot("one")["phase"] == "review"
    assert notices.snapshot("two") is None
    now[0] += 8
    assert notices.snapshot("one") is None


def test_rejected_or_error_is_not_a_verified_warning():
    notices = ConcealmentNotices()
    for confirmed, errored in [(False, False), (True, True)]:
        a = alert()
        notices.candidate(a)
        notices.verdict(a, SimpleNamespace(confirmed=confirmed, errored=errored))
        assert notices.snapshot("one") is None


def test_old_rejection_cannot_clear_a_newer_review():
    notices = ConcealmentNotices()
    a, b = alert(), alert()
    notices.candidate(a)
    notices.candidate(b)
    notices.verdict(b, SimpleNamespace(confirmed=True))
    notices.verdict(a, SimpleNamespace(confirmed=False))
    assert notices.snapshot("one")["id"] == b.payload["concealment_notice_id"]
    notices.candidate(alert())
    assert notices.snapshot("one")["phase"] == "review"


def test_other_detectors_do_not_generate_concealment_notices():
    notices = ConcealmentNotices()
    notices.candidate(alert(detector="fire_smoke"))
    assert notices.snapshot("one") is None
