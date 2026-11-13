import unittest

from quantbullet.reporting.slides import (
    AgendaSlide,
    ContentSlide,
    RenderContext,
    SectionSlide,
    Slide,
    TakeawaySlide,
    TwoColumnSlide,
    get_layout,
    list_layouts,
    register_layout,
)


class TestSlides(unittest.TestCase):
    def setUp(self):
        self.ctx = RenderContext()

    def test_heading_attributes_and_notes(self):
        slide = ContentSlide("Results\nQ3", classes=["smaller"], background="#ffffff", id="res")
        slide.text("body").speaker_notes("say this")
        md = slide.to_markdown(self.ctx, extra_classes=["appendix"])
        self.assertTrue(md.startswith('## Results Q3 {#res .smaller .appendix background-color="#ffffff"}'))
        self.assertIn("::: {.notes}\nsay this\n:::", md)

    def test_section_slide(self):
        md = SectionSlide("Overview", description="Rates").to_markdown(self.ctx)
        self.assertTrue(md.startswith("# Overview\n"))
        self.assertIn("::: {.section-desc}\nRates\n:::", md)
        self.assertIn("{.appendix-section}", SectionSlide("Appendix", appendix=True).to_markdown(self.ctx))

    def test_takeaway_without_title(self):
        md = TakeawaySlide("One message").to_markdown(self.ctx)
        self.assertTrue(md.startswith("## {.takeaway}"))
        self.assertIn("::: {.takeaway-message}\nOne message\n:::", md)

    def test_two_column_extra_items_go_below(self):
        slide = TwoColumnSlide("T", widths=(30, 70))
        slide.left.text("L")
        slide.right.text("R")
        slide.source("S")
        md = slide.to_markdown(self.ctx)
        self.assertLess(md.index("::::"), md.index("qb-source"))

    def test_agenda_uses_section_titles(self):
        self.ctx.section_titles = ["One", "Two"]
        self.assertIn("1. One\n2. Two", AgendaSlide().to_markdown(self.ctx))
        self.assertIn("1. X", AgendaSlide(items=["X"]).to_markdown(self.ctx))

    def test_register_custom_layout(self):
        @register_layout("scorecard-test")
        class ScorecardSlide(ContentSlide):
            def __init__(self, title=None, scores=None, **kwargs):
                super().__init__(title, **kwargs)
                self.kpis([dict(value=f"{v:.1f}", label=k) for k, v in scores.items()])

        self.assertIs(get_layout("scorecard-test"), ScorecardSlide)
        self.assertIn("scorecard-test", list_layouts())
        md = ScorecardSlide("Q3", scores={"a": 1}).to_markdown(self.ctx)
        self.assertTrue(md.startswith("## Q3 {.scorecard-test}"))
        with self.assertRaises(TypeError):
            register_layout("bad")(int)

    def test_unknown_layout(self):
        with self.assertRaises(KeyError):
            get_layout("nope")

    def test_container_rejects_non_components(self):
        with self.assertRaises(TypeError):
            Slide("x").add("text")


if __name__ == "__main__":
    unittest.main()
