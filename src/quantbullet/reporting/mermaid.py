"""
Render Mermaid diagram files to SVG or PDF.

Mermaid parses and lays out diagrams in JavaScript, so the rendering runs mermaid.js
inside a headless browser through Playwright. Playwright needs a browser: either run
``playwright install chromium`` once, or pass ``browser_channel="msedge"`` (or
``"chrome"``) to use a browser that is already installed.
"""
import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MERMAID_JS_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js"

_FORMATS = (".svg", ".pdf")

# Returns {svg} or {error}: a syntax error in the diagram is a normal outcome, not a crash.
_RENDER_JS = """
async ([source, config]) => {
    mermaid.initialize({ startOnLoad: false, ...config });
    try {
        const { svg } = await mermaid.render("mermaid-diagram", source);
        return { svg };
    } catch (err) {
        return { error: String(err && err.message ? err.message : err) };
    }
}
"""

# Puts the SVG on the page at its natural size and serializes it as XML. Mermaid's own
# string can hold HTML-only markup such as <br> inside labels, which is not valid SVG.
_PLACE_JS = """
([svgText, background]) => {
    document.body.innerHTML = svgText;
    const svg = document.body.querySelector("svg");
    const box = svg.viewBox.baseVal;
    svg.setAttribute("width", box.width);
    svg.setAttribute("height", box.height);
    svg.style.maxWidth = "none";
    svg.style.display = "block";
    svg.style.backgroundColor = background;
    return {
        width: box.width,
        height: box.height,
        svg: new XMLSerializer().serializeToString(svg),
    };
}
"""


def render_mermaid(
    source: str | Path,
    output: str | Path,
    *,
    theme: str = "default",
    background: str = "white",
    config: dict | None = None,
    browser_channel: str | None = None,
    mermaid_js: str | Path = MERMAID_JS_URL,
) -> Path:
    """Render a Mermaid diagram file to SVG or PDF.

    Parameters
    ----------
    source : str or Path
        Path to a file holding Mermaid diagram text (for example ``flow.mmd``).
    output : str or Path
        File to write. The suffix picks the format: ``.svg`` or ``.pdf``. A PDF has a
        single page sized to the diagram. Missing parent folders are created.
    theme : str
        Mermaid theme: ``"default"``, ``"neutral"``, ``"dark"``, ``"forest"`` or ``"base"``.
    background : str
        CSS background color, for example ``"white"`` or ``"transparent"``.
    config : dict, optional
        Extra ``mermaid.initialize`` settings, merged over ``theme``. For example
        ``{"htmlLabels": False}`` writes labels as plain SVG text instead of embedded
        HTML, which SVG tools other than browsers (Inkscape, cairosvg) can display.
    browser_channel : str, optional
        ``None`` uses Playwright's own Chromium (needs ``playwright install chromium``);
        ``"msedge"`` or ``"chrome"`` uses an installed Edge or Chrome instead.
    mermaid_js : str or Path
        URL or local file path of the mermaid.js bundle (``dist/mermaid.min.js``).
        The default loads a pinned version from a CDN; pass a local copy to work offline.

    Returns
    -------
    Path
        The written output file.

    Raises
    ------
    ValueError
        If the output suffix is not ``.svg`` or ``.pdf``, or Mermaid cannot parse the diagram.
    """
    source = Path(source)
    output = Path(output)
    if output.suffix.lower() not in _FORMATS:
        raise ValueError(f"output must end in .svg or .pdf, got {output.name!r}")
    text = source.read_text(encoding="utf-8")
    mermaid_config = {"theme": theme, **(config or {})}

    args = (text, output, mermaid_config, background, browser_channel, mermaid_js, source.name)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _render(*args)
    else:
        # Playwright's sync API refuses to run inside an event loop (Jupyter has one),
        # so run it on a worker thread there.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_render, *args).result()
    return output


def _render(text, output, mermaid_config, background, browser_channel, mermaid_js, name):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=browser_channel)
        try:
            page = browser.new_page()
            page.set_content("<!doctype html><html><body style='margin:0'></body></html>")
            if str(mermaid_js).startswith(("http://", "https://")):
                page.add_script_tag(url=str(mermaid_js))
            else:
                page.add_script_tag(path=str(mermaid_js))

            rendered = page.evaluate(_RENDER_JS, [text, mermaid_config])
            if "error" in rendered:
                raise ValueError(f"Mermaid could not render {name}: {rendered['error']}")
            placed = page.evaluate(_PLACE_JS, [rendered["svg"], background])

            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() == ".svg":
                output.write_text(placed["svg"], encoding="utf-8")
            else:
                page.evaluate("bg => { document.body.style.background = bg; }", background)
                page.pdf(
                    path=str(output),
                    width=f"{math.ceil(placed['width'])}px",
                    height=f"{math.ceil(placed['height'])}px",
                    print_background=True,
                    page_ranges="1",
                )
        finally:
            browser.close()
