"""Inventory fallback must not retry ambiguous allocation failures."""
import pytest

from netcast_tennisvision.cloud.ppio_lifecycle import (
    AnalysisCancelled,
    CloudLifecycleError,
    PPIOJobManager,
    PPIOLiveJobManager,
)


@pytest.fixture(params=[PPIOJobManager, PPIOLiveJobManager])
def lifecycle(request, tmp_path, monkeypatch):
    monkeypatch.setenv("TENNISVISION_PPIO_PRODUCT_ID", "L40S.22c125g")
    monkeypatch.setenv("TENNISVISION_PPIO_FALLBACK_PRODUCTS", "L40S.28c125g,4090.16c125g")
    m = (request.param(tmp_path, tmp_path / "status.json") if request.param is PPIOJobManager
         else request.param(tmp_path))
    monkeypatch.setattr(m, "_instance_envs", lambda: [])
    monkeypatch.setattr(m, "_rootfs_size_for_product", lambda: 60)
    return m


def test_inventory_walk_stops_at_first_success_and_resets_next_job(lifecycle, monkeypatch):
    attempts = []
    def provider(method, path, payload):
        attempts.append(payload["productId"])
        if payload["productId"] != "4090.16c125g":
            raise CloudLifecycleError("PPIO 请求失败：resource insufficient with product " + payload["productId"])
        return {"instanceId": "only-instance"}
    monkeypatch.setattr(lifecycle, "_provider_request", provider)
    lifecycle._write_json(lifecycle.status_path, {"session_id": "live-session", "state": "preparing"})
    assert lifecycle._create_instance() == "only-instance"
    assert attempts == ["L40S.22c125g", "L40S.28c125g", "4090.16c125g"]
    assert lifecycle.selected_product_id == "4090.16c125g"
    assert lifecycle.product_id == "L40S.22c125g"
    assert lifecycle._read_json(lifecycle.status_path)["session_id"] == "live-session"
    lifecycle._create_instance()
    assert attempts[3] == "L40S.22c125g"


@pytest.mark.parametrize("error", ["unauthorized", "insufficient balance", "image not found",
                                   "connection timed out", "invalid product", "quota exceeded"])
def test_non_inventory_failure_never_allocates_again(lifecycle, monkeypatch, error):
    attempts = []
    def provider(*args):
        attempts.append(args)
        raise CloudLifecycleError(error)
    monkeypatch.setattr(lifecycle, "_provider_request", provider)
    with pytest.raises(CloudLifecycleError, match=error):
        lifecycle._create_instance()
    assert len(attempts) == 1


def test_missing_instance_id_is_not_inventory_failure(lifecycle, monkeypatch):
    attempts = []
    def provider(*args):
        attempts.append(args)
        return {"success": True}
    monkeypatch.setattr(lifecycle, "_provider_request", provider)
    with pytest.raises(CloudLifecycleError, match="实例编号"):
        lifecycle._create_instance()
    assert len(attempts) == 1


def test_all_unavailable_is_bounded_and_reports_candidates(lifecycle, monkeypatch):
    monkeypatch.setattr(lifecycle, "_provider_request", lambda *args: (_ for _ in ()).throw(
        CloudLifecycleError("resource insufficient with product test")))
    with pytest.raises(CloudLifecycleError, match="均无库存"):
        lifecycle._create_instance()
    assert len(lifecycle.gpu_attempts) == 3
    assert lifecycle.selected_product_id is None


def test_stop_between_candidates_prevents_next_allocation(lifecycle, monkeypatch):
    def provider(*args):
        lifecycle._cancel_event.set()
        raise CloudLifecycleError("resource insufficient with product test")
    monkeypatch.setattr(lifecycle, "_provider_request", provider)
    with pytest.raises(AnalysisCancelled):
        lifecycle._create_instance()
    assert len(lifecycle.gpu_attempts) == 1


def test_rootfs_retry_can_then_fall_back(lifecycle, monkeypatch):
    attempts = []
    def provider(method, path, payload):
        attempts.append((payload["productId"], payload["rootfsSize"]))
        if len(attempts) == 1:
            raise CloudLifecycleError("rootfs size must not be more than 42 GB")
        if len(attempts) == 2:
            raise CloudLifecycleError("resource insufficient with product test")
        return {"instanceId": "success"}
    monkeypatch.setattr(lifecycle, "_provider_request", provider)
    assert lifecycle._create_instance() == "success"
    assert attempts == [("L40S.22c125g", 60), ("L40S.22c125g", 42), ("L40S.28c125g", 60)]
