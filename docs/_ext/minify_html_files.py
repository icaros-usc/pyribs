"""Extension for minifying content.

Requires minify-html (https://github.com/wilsonzlin/minify-html) or beautifulsoup4 to be
installed.
"""

import contextlib
from pathlib import Path

with contextlib.suppress(ImportError):
    import minify_html

with contextlib.suppress(ImportError):
    from bs4 import BeautifulSoup

from sphinx.util import logging

logger = logging.getLogger(__name__)


def minify_html_files(app, exception):
    """Minifies all HTML files after build is done."""
    if exception is not None:
        return  # Build failed, skip.

    if app.config.minify_mode is None:
        logger.info("Skipping minifying")
        return

    outdir = Path(app.outdir)
    html_files = list(outdir.rglob("*.html"))
    logger.info(f"Minifying {len(html_files)} HTML files...")

    for path in html_files:
        try:
            text = path.read_text(encoding="utf-8")
            if app.config.minify_mode == "minify":
                minified = minify_html.minify(text, minify_js=True, minify_css=True)
            elif app.config.minify_mode == "prettify":
                minified = BeautifulSoup(text, "html.parser").prettify()
            path.write_text(minified, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to minify {path}: {e}")


def setup(app):
    """Installs the extension."""
    app.add_config_value("minify_mode", None, "html")
    app.connect("build-finished", minify_html_files)
