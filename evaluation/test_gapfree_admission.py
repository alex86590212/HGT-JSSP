import csv
import subprocess
import sys
from pathlib import Path


def test_exchange_cli_reports_uncapped_search_diagnostics(tmp_path):
    output = tmp_path / "exchange_results.csv"
    result = subprocess.run(
        [
            sys.executable, "-m", "evaluation.eval_gapfree_admission",
            "--rates", "1.2", "--seeds", "0", "--n-scenarios", "1",
            "--episode-duration", "12",
            "--methods", "edf_commitment_exchange",
            "arrival_commitment_exchange", "gain_commitment_exchange",
            "--output-csv", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    with output.open() as stream:
        row, = csv.DictReader(stream)
    assert float(row["n_veh_avg"]) > 0
    assert not any("budget" in column for column in row)
    for name in ("edf_commitment_exchange", "arrival_commitment_exchange", "gain_commitment_exchange"):
        evaluated = float(row[f"{name}_evaluated_mean"])
        proposed = float(row[f"{name}_proposed_mean"])
        profitable = float(row[f"{name}_profitable_mean"])
        accepted = float(row[f"{name}_accepted_mean"])
        assert evaluated > 0
        assert evaluated >= proposed >= profitable >= accepted >= 0
        assert float(row[f"{name}_planned_gain_mean"]) >= 0
        assert float(row[f"{name}_runtime_seconds_mean"]) > 0
        assert float(row[f"{name}_wt_mean"]) >= 0
        assert 0 <= float(row[f"{name}_comp_mean"]) <= 1
