from nutriscan.core.state_machine import NutriScanStateMachine, ScanState


def test_state_machine_full_happy_path_and_guards():
    sm = NutriScanStateMachine()
    assert sm.state == ScanState.IDLE

    # Cannot finish processing before triggering
    assert sm.finish_processing({"scan_id": "x"}) is False
    assert sm.complete_cooldown() is False

    # IDLE -> STABILIZING
    assert sm.begin_stabilizing(120.0) is True
    assert sm.state == ScanState.STABILIZING
    assert sm.begin_stabilizing(130.0) is False

    # Abort stabilizing back to IDLE
    assert sm.abort_stabilizing() is True
    assert sm.state == ScanState.IDLE

    # IDLE -> STABILIZING -> PROCESSING -> SENDING -> COOLDOWN -> IDLE
    assert sm.begin_stabilizing(350.0) is True
    assert sm.trigger_processing(350.0) is True
    assert sm.state == ScanState.PROCESSING

    payload = {"scan_id": "abc", "berat_total_g": 350.0}
    assert sm.finish_processing(payload, has_occlusion=True) is True
    assert sm.state == ScanState.SENDING

    snap = sm.snapshot()
    assert snap.has_occlusion is True
    assert snap.locked_weight_g == 350.0
    assert snap.last_scan_payload == payload

    assert sm.finish_sending() is True
    assert sm.state == ScanState.COOLDOWN

    assert sm.complete_cooldown() is True
    assert sm.state == ScanState.IDLE
    assert sm.snapshot().locked_weight_g is None


def test_state_machine_error_and_empty_detection_transitions():
    sm = NutriScanStateMachine()
    sm.trigger_processing(200.0)
    assert sm.skip_empty_processing() is True
    assert sm.state == ScanState.COOLDOWN
    assert sm.complete_cooldown() is True

    sm.trigger_processing(210.0)
    assert sm.cancel_processing("kamera putus") is True
    assert sm.state == ScanState.COOLDOWN
    assert sm.snapshot().last_error == "kamera putus"
