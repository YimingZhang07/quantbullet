import asyncio
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from pypdf import PdfReader

from quantbullet.reporting import render_mermaid
from tests.artifacts import artifact_dir

FLOWCHART = """flowchart LR
    A[Raw loans] --> B{Clean?}
    B -- yes --> C[Fit model<br/>GAM]
    B -- no --> D[Fix data]
    D --> B
"""

SVG_NS = "{http://www.w3.org/2000/svg}"


def find_browser_channel() -> str | None:
    """Return a launchable Playwright channel: its own Chromium, else installed Edge or Chrome."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as p:
        for channel in (None, "msedge", "chrome"):
            try:
                p.chromium.launch(channel=channel).close()
                return channel
            except Error:
                continue
    raise unittest.SkipTest("no browser for Playwright; run `playwright install chromium`")


class TestRenderMermaid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.channel = find_browser_channel()

    def setUp(self):
        self.out_dir = Path(artifact_dir(self, "reporting/mermaid"))
        self.source = self.out_dir / "flow.mmd"
        self.source.write_text(FLOWCHART, encoding="utf-8")

    def render(self, output: str, source: Path | None = None, **kwargs) -> Path:
        return render_mermaid(source or self.source, self.out_dir / output,
                              browser_channel=self.channel, **kwargs)

    def test_svg_is_standalone_xml_with_explicit_size(self):
        path = self.render("flow.svg")
        root = ET.parse(path).getroot()

        self.assertEqual(root.tag, f"{SVG_NS}svg")
        self.assertGreater(float(root.get("width")), 0)
        self.assertGreater(float(root.get("height")), 0)
        text = path.read_text(encoding="utf-8")
        for label in ("Raw loans", "Clean?", "Fix data"):
            self.assertIn(label, text)

    def test_pdf_is_one_page_sized_to_the_diagram(self):
        svg_root = ET.parse(self.render("flow.svg")).getroot()
        path = self.render("nested/flow.pdf")

        pages = PdfReader(path).pages
        self.assertEqual(len(pages), 1)
        # CSS px to PDF points is 0.75; the page is the SVG size rounded up to whole px.
        self.assertAlmostEqual(float(pages[0].mediabox.width),
                               float(svg_root.get("width")) * 0.75, delta=1.0)
        self.assertIn("Raw loans", pages[0].extract_text())

    def test_config_html_labels_off_gives_plain_svg_text(self):
        default = self.render("html_labels.svg").read_text(encoding="utf-8")
        plain = self.render("plain_labels.svg", config={"htmlLabels": False}).read_text(encoding="utf-8")

        self.assertIn("<foreignObject", default)
        self.assertNotIn("<foreignObject", plain)
        # Plain SVG text puts each word in its own <tspan>, so look for a one-word label.
        self.assertTrue(">Clean?</tspan>" in plain)

    def test_syntax_error_raises_value_error_naming_the_file(self):
        bad = self.out_dir / "bad.mmd"
        bad.write_text("flowchart LR\n    A --> B -->\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "bad.mmd"):
            self.render("bad.svg", source=bad)
        self.assertFalse((self.out_dir / "bad.svg").exists())

    def test_unsupported_suffix_raises_before_rendering(self):
        with self.assertRaisesRegex(ValueError, r"\.svg or \.pdf"):
            self.render("flow.png")

    def test_runs_inside_an_event_loop(self):
        # Jupyter runs code inside an event loop, where Playwright's sync API would refuse.
        async def render_in_loop():
            return self.render("in_loop.svg")

        self.assertTrue(asyncio.run(render_in_loop()).exists())


if __name__ == "__main__":
    unittest.main()
