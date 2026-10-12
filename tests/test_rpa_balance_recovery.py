import json
from pathlib import Path

import pytest

from comparisons.search_exploration.recover_rpa_directional_balance import (
    balance_error, replace_output, request_audit)


def test_only_provider_balance_errors_trigger_recovery():
    assert balance_error('dsh: QUOTA: Insufficient Balance')
    assert balance_error('insufficient_balance')
    assert not balance_error('429: Too many requests')
    assert not balance_error('Empty candidate payload')


def test_recovery_preserves_scientific_command_and_original():
    command = ['python', '-m', 'runner', '--output', '/old', '--seed', '16']
    new = replace_output(command, Path('/new'))
    assert new == ['python', '-m', 'runner', '--output', '/new', '--seed', '16']
    assert command[4] == '/old'
    with pytest.raises(ValueError):
        replace_output(['python'], Path('/new'))


def test_history_and_uncollected_stderr_are_both_audited(tmp_path):
    folder = tmp_path / 'harness-workspaces/w/calls/0001'
    folder.mkdir(parents=True)
    stderr = folder / 'stderr.txt'
    stderr.write_text('dsh: QUOTA: Insufficient Balance')
    assert request_audit(tmp_path)['balance_affected']
    (tmp_path / 'llm_history.json').write_text(json.dumps([
        {'error': f'DeepSeek Harness failed; inspect {stderr} (exit code 1).'},
        {'valid': 10}, {'error': 'Empty candidate payload'}]))
    a = request_audit(tmp_path)
    assert a['failed_requests'] == 2
    assert a['balance_failed_requests'] == 1
    assert a['valid_members'] == 10


def test_normal_model_failure_is_retained_not_rerun(tmp_path):
    (tmp_path / 'llm_history.json').write_text(json.dumps([{'error': 'JSON parse failed'}]))
    a = request_audit(tmp_path)
    assert a['failed_requests'] == 1
    assert not a['balance_affected']
