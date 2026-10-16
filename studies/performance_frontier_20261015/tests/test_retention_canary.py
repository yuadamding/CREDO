import pytest

from frontier.retention_canary import run_canary


def test_literal_canary_covers_real_retirement_without_training(tmp_path):
    result = run_canary(tmp_path/'canary')
    assert result['status']=='passed'
    assert result['optimizer_calls']==result['scientific_updates']==result['real_checkpoint_reads']==0
    assert result['corrected_retirement']['payload_absence_verified'] is True
    assert result['legacy_observation']['pathname_absent_after_close'] is True
    assert result['gpu_runtime_qualified'] is False
    assert len(list((tmp_path/'canary').rglob('checkpoint.pt')))==4
    assert sum(p.stat().st_size for p in (tmp_path/'canary').rglob('*') if p.is_file()) < 1024**2
    with pytest.raises(FileExistsError):
        run_canary(tmp_path/'canary')
