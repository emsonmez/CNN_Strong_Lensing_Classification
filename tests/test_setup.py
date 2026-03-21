import runpy
from unittest import mock


def test_setup_runs_without_error():
    """
    Ensure setup.py executes without errors.

    This mocks setuptools.setup so no real installation occurs,
    but still executes the file for coverage.
    """

    with mock.patch("setuptools.setup") as mock_setup:
        runpy.run_path("setup.py")

        # Ensure setup() was called
        assert mock_setup.called
