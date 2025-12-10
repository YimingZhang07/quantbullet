import unittest
import pandas as pd
import shutil
from pathlib import Path
from quantbullet.reporting.excel_exporter import ExcelExporter
from quantbullet.reporting.columns import ColumnFormat, ColumnMeta, ColumnSchema
from quantbullet.reporting.pdf_text_report import PdfTextReport, PdfColumnFormat, PdfColumnMeta

class TestExcelExporter(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        # shutil.rmtree(self.cache_dir, ignore_errors=True)
        # Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4.5678, 5.6789, 6.7890],
            "C": ["foo", "bar", "baz"],
            "D": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "E": ["abc", "def", "ghi"]
        })

        df["D"] = pd.to_datetime( df["D"] )
        self.df = df
        return df
    
    def tearDown(self):
        # shutil.rmtree(self.cache_dir, ignore_errors=True)
        # Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        pass

    def test_excel_exporter_basic(self):
        df = self.df
        exporter = ExcelExporter( filename=str( Path(self.cache_dir) / "test_exporter.xlsx" ) )
        schema = ColumnSchema( columns=[
            ColumnMeta( name="A", display_name="Column A", format=ColumnFormat( decimals=0 ) ),
            ColumnMeta( name="B", display_name="Column B", format=ColumnFormat( decimals=2, comma=True, transformer=lambda x: x * 1000 ) ),
            ColumnMeta( name="C", display_name="Column C", format=ColumnFormat() ),
            ColumnMeta( name="D", display_name="Date Column", format=ColumnFormat() )
        ] )
        exporter.add_sheet( sheet_name="TestSheet", df = df, schema=schema )
        exporter.save()
        
    def test_pdf_text_report( self ):
        df = self.df
        report = PdfTextReport( file_path=str( Path(self.cache_dir) / "test_report.pdf" ) )
        schema = [
            PdfColumnMeta(name="A", display_name="Column A", format=PdfColumnFormat(decimals=0)),
            PdfColumnMeta(name="B", display_name="Column B", format=PdfColumnFormat(decimals=2, comma=True, transformer=lambda x: x * 1000)),
            PdfColumnMeta(name="C", display_name="Column C", format=PdfColumnFormat()),
            PdfColumnMeta(name="D", display_name="Date Column", format=PdfColumnFormat())
        ]
        report.add_df_table( df=df, schema=schema )
        report.save()

        # test convert the pdf schema to excel schema
        def convert_pdf_schema_to_excel_schema( pdf_schema ):
            columns = []
            for col in pdf_schema:
                columns.append( ColumnMeta( name=col.name, display_name=col.display_name, format=ColumnFormat(
                    decimals=col.format.decimals,
                    comma=col.format.comma,
                    percent=col.format.percent,
                    transformer=col.format.transformer,
                ) ) )
            return ColumnSchema( columns=columns )
        
        excel_schema = convert_pdf_schema_to_excel_schema( schema )
        exporter = ExcelExporter( filename=str( Path(self.cache_dir) / "test_report_converted.xlsx" ) )
        exporter.add_sheet( sheet_name="TestSheet", df = df, schema=excel_schema )
        exporter.save()