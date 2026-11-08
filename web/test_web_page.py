"""The JavaScript of the interactive page web/numerical-methods.html gives the same numbers as the weekly modules."""
import importlib.util
import shutil
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is needed to run the page's JavaScript")
def test_page_matches_weekly_modules():
    spec = importlib.util.spec_from_file_location("verify_web_page", Path(__file__).with_name("verify_web_page.py"))
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    worst = verify.run()
    assert max(worst.values()) <= verify.REL_TOL, worst
