import subprocess
import sys


def test_trainable_parameter_guard_accepts_expected_count(tmp_path):
    log_file = tmp_path / "train.log"
    result = subprocess.run(
        [
            sys.executable,
            "verified_lora_train.py",
            "--expected-millions",
            "32.464896",
            "--log-file",
            str(log_file),
            "--",
            sys.executable,
            "-c",
            "print('#trainable params: 32.464896 M'); print('Starting training')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "#trainable params: 32.464896 M" in log_file.read_text()
    assert "Starting training" in log_file.read_text()


def test_trainable_parameter_guard_rejects_resume_shape():
    result = subprocess.run(
        [
            sys.executable,
            "verified_lora_train.py",
            "--expected-millions",
            "32.464896",
            "--",
            sys.executable,
            "-c",
            "print('#trainable params: 366.9 M'); print('Starting training')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "TRAINABLE-PARAMETER GUARD FAILED" in result.stderr
