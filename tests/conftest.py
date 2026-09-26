import pytest


def _is_manual(item: pytest.Item) -> bool:
    return item.get_closest_marker("manual") is not None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip manual triggers unless this run selected only those triggers.

    A full suite still collects them, so they stay visible in the Test panel.
    Clicking one test's run button collects only that test, and it executes.
    """
    if items and all(_is_manual(item) for item in items):
        return

    skip_manual = pytest.mark.skip(
        reason="manual trigger; use the run button on this test"
    )
    for item in items:
        if _is_manual(item):
            item.add_marker(skip_manual)
