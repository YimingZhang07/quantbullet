import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantbullet.reporting.slides import DEFAULT_THEME, SlideTheme  # noqa: E402


class TestSlideTheme(unittest.TestCase):
    def test_rejects_non_hex_colors(self):
        with self.assertRaises(ValueError):
            SlideTheme(primary="navy")
        with self.assertRaises(ValueError):
            SlideTheme(palette=("#000000", "red"))

    def test_palette_defaults_to_tokens(self):
        theme = SlideTheme()
        self.assertEqual(theme.colors[:2], (theme.primary, theme.accent))
        self.assertEqual(SlideTheme(palette=["#111111"]).colors, ("#111111",))

    def test_replace_returns_new_theme(self):
        dark = DEFAULT_THEME.replace(primary="#141f52")
        self.assertEqual(dark.primary, "#141f52")
        self.assertEqual(DEFAULT_THEME.primary, "#0b3d6e")

    def test_scss_has_every_token(self):
        scss = SlideTheme(accent="#e3120b").to_scss()
        self.assertIn("$qb-accent: #e3120b;", scss)
        self.assertIn("$qb-bg-alt:", scss)
        self.assertIn('$qb-font-family: "Source Sans 3"', scss)

    def test_bundled_font_files(self):
        self.assertEqual(set(DEFAULT_THEME.font_files()), {"Regular", "Bold", "Italic"})
        self.assertEqual(SlideTheme(font_family="Some System Font").font_files(), {})

    def test_matplotlib_context_sets_cycle(self):
        with DEFAULT_THEME.matplotlib():
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            self.assertEqual(colors[0], DEFAULT_THEME.primary)
        self.assertNotEqual(plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                            DEFAULT_THEME.primary)

    def test_style_plotly_keeps_figure_settings(self):
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(y=[1, 2]))
        fig.update_layout(margin=dict(l=5))
        DEFAULT_THEME.style_plotly(fig)
        self.assertEqual(list(fig.layout.template.layout.colorway)[0], DEFAULT_THEME.primary)
        self.assertEqual(fig.layout.margin.l, 5)

    def test_style_plotly_swaps_express_default_colors(self):
        import plotly.express as px

        fig = px.line(x=[1, 2, 1, 2], y=[1, 2, 3, 4], color=["a", "a", "b", "b"])
        fig.data[1].line.color = "#123456"  # set by hand: kept
        DEFAULT_THEME.style_plotly(fig)
        self.assertEqual(fig.data[0].line.color, DEFAULT_THEME.colors[0])
        self.assertEqual(fig.data[1].line.color, "#123456")

    def test_plotly_context_sets_default_template(self):
        import plotly.express as px
        import plotly.io as pio

        before = pio.templates.default
        with DEFAULT_THEME.plotly():
            fig = px.line(x=[1, 2], y=[1, 2])
        self.assertEqual(fig.data[0].line.color, DEFAULT_THEME.primary)
        self.assertEqual(pio.templates.default, before)


if __name__ == "__main__":
    unittest.main()
