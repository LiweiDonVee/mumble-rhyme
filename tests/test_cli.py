import json
import subprocess
import sys


def test_cli_four_b_status_selects_model_without_loading_it():
    result = subprocess.run(
        [sys.executable, "-m", "mumble_rhyme", "--model", "4b", "status"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["model_id"] == "Qwen/Qwen3.5-4B"
    assert status["loaded"] is False
