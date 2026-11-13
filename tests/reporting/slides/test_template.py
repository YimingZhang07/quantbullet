import json
import tempfile
import unittest
from pathlib import Path

from quantbullet.reporting.slides import (
    SlideTemplate,
    SlideTheme,
    export_extension,
    export_template,
    get_template,
    list_templates,
    register_template,
)


class TestTemplates(unittest.TestCase):
    def test_builtin_default(self):
        template = get_template("default")
        self.assertIn("default", list_templates())
        self.assertEqual(template.format_name, "qb-default-revealjs")
        self.assertIn("disclaimer", template.partials)
        self.assertIn("## Disclaimer", template.partial("disclaimer"))

    def test_unknown_names(self):
        with self.assertRaises(KeyError):
            get_template("no-such-template")
        with self.assertRaises(KeyError):
            get_template("default").partial("nope")

    def test_install_writes_theme_tokens_and_fonts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = get_template("default").install(tmp, SlideTheme(primary="#123456"))
            self.assertEqual(dest, Path(tmp) / "_extensions" / "qb-default")
            scss = (dest / "theme.scss").read_text(encoding="utf-8")
            defaults = scss.index("/*-- scss:defaults --*/")
            self.assertLess(defaults, scss.index("$qb-primary: #123456;"))
            self.assertLess(scss.index("$qb-primary: #123456;"), scss.index("$qb-primary: #0b3d6e !default;"))
            self.assertIn("@font-face", (dest / "fonts.html").read_text(encoding="utf-8"))
            self.assertTrue((dest / "_extension.yml").is_file())

    def test_export_register_and_use_a_custom_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = export_template("default", Path(tmp) / "acme")
            meta = json.loads((source / "template.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["name"], meta["extension"]), ("acme", "qb-acme"))
            template = SlideTemplate.from_dir(source, theme=SlideTheme(accent="#e3120b"))
            register_template(template, replace=True)
            self.assertIs(get_template("acme"), template)
            self.assertEqual(template.format_name, "qb-acme-revealjs")
            with self.assertRaises(ValueError):
                register_template(template)
            with self.assertRaises(FileExistsError):
                export_template("default", source)

    def test_from_dir_needs_extension_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                SlideTemplate.from_dir(tmp)

    def test_export_extension_for_handwritten_decks(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = export_extension("default", tmp)
            self.assertTrue((dest / "sections.lua").is_file())


if __name__ == "__main__":
    unittest.main()
