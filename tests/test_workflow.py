import csv
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

from logger import SessionLogger
from trial_loop import ExperimentRunner, check_phase_switch


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_accuracy_threshold_controls_reinforcement_phases(self):
        self.assertEqual(
            check_phase_switch(15, 15, [True] * 8 + [False] * 2, 100, False, 15, 10, 0.8),
            (True, "accuracy_met"),
        )
        self.assertEqual(
            check_phase_switch(15, 15, [True] * 7 + [False] * 3, 100, False, 15, 10, 0.8),
            (False, ""),
        )
        self.assertEqual(
            check_phase_switch(100, 0, [False] * 10, 100, True, 15, 10, 0.8),
            (True, "max_trials"),
        )

    def test_presentation_only_includes_lists_loaded_for_group(self):
        runner = object.__new__(ExperimentRunner)
        runner.cfg = {
            "phases": [
                {"reinforced_list": 1, "extinction_lists": []},
                {"reinforced_list": 2, "extinction_lists": [1]},
                {"reinforced_list": None, "extinction_lists": [1, 2, 3]},
            ]
        }
        runner.list_manager = type("Lists", (), {"full_lists": {1: set(), 2: set()}})()
        self.assertEqual(runner._presentation_list_numbers(), [1, 2])
        runner.list_manager.full_lists[3] = set()
        self.assertEqual(runner._presentation_list_numbers(), [1, 2, 3])

    def test_survey_submission_is_filtered_and_unblocks_waiter(self):
        runner = object.__new__(ExperimentRunner)
        runner.cfg = {
            "survey": {
                "questions": [
                    {"id": "age"},
                    {"id": "race_ethnicity"},
                ]
            }
        }
        runner.running = True
        runner._survey_waiting = True
        runner._survey_event = threading.Event()
        runner._survey_responses = None

        accepted = runner.submit_survey({
            "age": " 21 ",
            "race_ethnicity": ["Asian", "White"],
            "unexpected": "not stored",
        })

        self.assertTrue(accepted)
        self.assertTrue(runner._survey_event.is_set())
        self.assertEqual(runner._survey_responses, {
            "age": "21",
            "race_ethnicity": ["Asian", "White"],
        })

    def test_logger_persists_and_exports_survey(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(str(Path(tmp) / "experiment.db"), "PTEST", 2)
            logger.log_survey_responses({
                "age": "21",
                "race_ethnicity": ["Asian", "White"],
                "nationality": "",
            })
            logger.close(export_csv=True)

            with closing(sqlite3.connect(logger.db_path)) as conn:
                rows = conn.execute(
                    "SELECT question_id, response FROM survey_responses "
                    "WHERE session_id=? ORDER BY id",
                    (logger.session_id,),
                ).fetchall()
            self.assertEqual(rows[0], ("age", "21"))
            self.assertEqual(json.loads(rows[1][1]), ["Asian", "White"])
            self.assertEqual(rows[2], ("nationality", ""))

            survey_csv = Path(tmp) / f"{logger.session_id}_survey.csv"
            self.assertTrue(survey_csv.exists())
            with survey_csv.open(encoding="utf-8-sig", newline="") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual([row["question_id"] for row in exported], [
                "age", "race_ethnicity", "nationality",
            ])

    def test_survey_precedes_thank_you_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({
                "experiment": {"group": 2, "participant_id": "PTEST"},
                "phases": [],
                "switching": {"max_trials": 1},
                "trial": {"window_seconds": 0.01, "iti_seconds": 0},
                "lists": {
                    "path": str(ROOT / "list"),
                    "list1": "list1_animals.txt",
                    "list2": "list2_professions.txt",
                    "list3": "list3_food.txt",
                },
                "audio": {
                    "reinforcement_sound": str(ROOT / "audio" / "reward.wav"),
                    "sample_rate": 16000,
                    "channels": 1,
                },
                "asr": {"language": "en"},
                "practice": {"enabled": False},
                "instructions": {"end": "Thank you"},
                "survey": {
                    "enabled": True,
                    "title": "Final Survey",
                    "intro": "",
                    "questions": [{"id": "age", "type": "number", "label": "Age"}],
                },
                "data": {"db_path": str(tmp_path / "experiment.db"), "export_csv": True},
            }), encoding="utf-8")

            runner = ExperimentRunner(str(config_path), model=object())
            events = []

            def on_survey(_survey):
                events.append("survey")
                self.assertTrue(runner.submit_survey({"age": "21"}))

            def on_instruction(text, _words, _countdown, _title):
                events.append("thank_you" if text == "Thank you" else "combined_rules")
                runner.acknowledge_instruction()

            runner.on_survey = on_survey
            runner.on_instruction = on_instruction
            runner.on_complete = lambda _summary: events.append("complete")
            runner.run()

            self.assertEqual(events, ["combined_rules", "survey", "thank_you", "complete"])
            self.assertFalse(runner.running)


if __name__ == "__main__":
    unittest.main()
