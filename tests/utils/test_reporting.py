import unittest
import pandas as pd
from datetime import date
from quantbullet.reporting import HTMLPageBuilder, HTMLTableBuilder


class TestHTMLBuilders(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Name'  : [ 'Alice', 'Bob' ],
            'Score' : [ 88.1234, 92.4567 ],
            'Status': [ 'pass', 'fail' ],
            'Date'  : [ date(2025, 1, 1), date(2025, 1, 2) ]
        })
        
        
        self.column_settings = {
            'Name'  : {'width': '10%', 'align': 'left'},
            'Score' : {'width': '10%', 'align': 'right', 'formatter': lambda x: f'{x:.1f}'},
            'Status': {'width': '10%', 'align': 'center', 'formatter': str.upper},
            'Date'  : {'width': '10%', 'align': 'center', 'formatter': lambda x: x.strftime('%Y-%m-%d')}
        }

    def test_build_html_page( self ):
        """A trivial test just to check if the HTML page builder works."""
        table_html = HTMLTableBuilder(
            self.df, 
            title="Test Results", 
            column_settings=self.column_settings,
            inline_style=True
        ).to_html()

        # Assemble page
        page = HTMLPageBuilder()
        page.add_heading("Quarterly Report", level=1)
        page.add_paragraph("The following tables summarize recent performance.")
        page.add_table(table_html)
        html_output = page.build()
        self.assertIn("<h1>Quarterly Report</h1>", html_output)
        # with open("preview.html", "w", encoding="utf-8") as f:
        #     f.write(html_output)

        # import webbrowser
        # webbrowser.open("preview.html")
