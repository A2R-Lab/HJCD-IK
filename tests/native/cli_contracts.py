"""End-to-end native CLI contracts; run through CTest, not the Python-wheel suite.

Negative input tests hide CUDA devices to verify validation precedes initialization.
Positive/output-failure tests require the same GPU as native_api. No timing assertions.
"""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv[1]).resolve()) if __name__ == "__main__" else None
HEADER = "target_id,target_px,target_py,target_pz,target_qx,target_qy,target_qz,target_qw"
ROW = "7,0.3,0.0,0.5,0,0,0,1"


class CliContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hjcd-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_cli(self, *args, gpu=False):
        env = dict(os.environ)
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            [BINARY, *args], cwd=self.root, env=env,
            capture_output=True, text=True, timeout=120,
        )

    def write_csv(self, body, header=HEADER):
        path = self.root / "targets.csv"
        path.write_bytes((header + "\n" + body).encode())
        return path

    def assert_failure(self, run, expected):
        self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn(expected, run.stderr)
        self.assertNotIn("[OK]", run.stdout)
        self.assertNotIn("[from_csv] Wrote", run.stdout)

    def test_help_without_gpu(self):
        run = self.run_cli("--help")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--mode=single|sweep|from_csv", run.stdout)

    def test_invalid_options_without_gpu(self):
        for option in ("--batch_size", "--num_solutions", "--num_targets"):
            for value in ("", "0", "-1", "1.5", "10garbage", "no", "999999999999999999"):
                with self.subTest(option=option, value=value):
                    self.assert_failure(self.run_cli(option + "=" + value), option)
        self.assert_failure(self.run_cli("--typo=1"), "Unknown argument")
        self.assert_failure(self.run_cli("--mode=typo"), "Unknown --mode")

    def test_missing_and_empty_csv(self):
        self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=missing.csv"), "Failed to open")
        path = self.root / "empty.csv"
        path.touch()
        self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path)), "Empty CSV")
        path = self.write_csv("")
        self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path)), "No targets found")

    def test_csv_schema(self):
        path = self.write_csv(ROW, header="target_id,target_px")
        self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path)), "missing required")
        for row in ("7,0.3", ROW + ",extra"):
            path = self.write_csv(row)
            self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path)),
                                "targets.csv:2: wrong number")

    def test_malformed_csv_numbers(self):
        for index, value in ((0, "7suffix"), (0, "99999999999999999"),
                             (1, "0.3suffix"), (2, "nan"), (3, "inf"),
                             (4, "1e999"), (7, "0")):
            with self.subTest(index=index, value=value):
                fields = ROW.split(",")
                fields[index] = value
                path = self.write_csv(",".join(fields))
                self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path)),
                                    "targets.csv:2:")

    def test_invalid_duplicate_row_not_hidden(self):
        path = self.write_csv(ROW + "\n" + ROW.replace("0.3", "broken"))
        self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path)),
                            "targets.csv:3: target_px")

    def test_output_open_failure(self):
        path = self.write_csv(ROW)
        self.assert_failure(self.run_cli("--mode=from_csv", "--csv_in=" + str(path),
                                        "--csv_out=missing/output.csv"), "Cannot open csv_out")

    def test_crlf_whitespace_duplicate_targets(self):
        path = self.write_csv("  " + ROW + " \r\n" + ROW + "\r\n\r\n", header=HEADER + "\r")
        out = self.root / "solutions.csv"
        run = self.run_cli("--mode=from_csv", "--batch_size=1",
                           "--csv_in=" + str(path), "--csv_out=" + str(out), gpu=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("from 1 targets", run.stdout)
        with out.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreater(len(rows), 0)
        self.assertEqual({row["target_id"] for row in rows}, {"7"})
        self.assertEqual(len({row["sample_id"] for row in rows}), len(rows))

    def test_yaml_output(self):
        out = self.root / "results.yml"
        run = self.run_cli("--batch_size=1", "--yaml_out=" + str(out), gpu=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        content = out.read_text()
        for key in ("Batch-Size", "IK-time(ms)", "Pos-Error", "Ori-Error"):
            self.assertIn(key + ":\n  - ", content)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_buffered_write_failures(self):
        self.assert_failure(self.run_cli("--batch_size=1", "--yaml_out=/dev/full", gpu=True),
                            "Failed to write output: /dev/full")
        path = self.write_csv(ROW)
        self.assert_failure(self.run_cli("--mode=from_csv", "--batch_size=1",
                                        "--csv_in=" + str(path), "--csv_out=/dev/full", gpu=True),
                            "Failed to write output: /dev/full")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
