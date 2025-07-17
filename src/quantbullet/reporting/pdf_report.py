
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import Tuple
import numpy as np

class PdfReport:
    def __init__(self, filepath: str, layout: Tuple[int, int], figsize: Tuple[int, int]=(11, 8.5)):
        """
        Initializes the PDF report creator.

        Parameters
        ----------
        filepath : str
            Path to save the PDF.
        layout : tuple of int
            (rows, cols) per page.
        figsize : tuple of int, optional
            Size of each PDF page in inches (default is (11, 8.5)).
        """
        self.filepath = filepath
        self.layout = layout
        self.figsize = figsize
        self.pdf = PdfPages(self.filepath)
        self.fig = None
        self.axes_list = []
        self.current_ax_idx = 0

    def _add_page(self):
        """
        Internal method to create a new page with axes.
        """
        self.fig, axes = plt.subplots(*self.layout, figsize=self.figsize)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        else:
            axes = axes.flatten()
        self.axes_list = axes
        self.current_ax_idx = 0

    def get_next_ax(self):
        """
        Gets the next available Axes object. Automatically handles pagination.
        """
        if self.fig is None or self.current_ax_idx >= len(self.axes_list):
            if self.fig is not None:
                self.pdf.savefig(self.fig, bbox_inches='tight')
                plt.close(self.fig)
            self._add_page()
        ax = self.axes_list[self.current_ax_idx]
        self.current_ax_idx += 1
        return ax

    def save(self):
        """
        Finalizes and saves the PDF report.
        """
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches='tight')
            plt.close(self.fig)
        self.pdf.close()