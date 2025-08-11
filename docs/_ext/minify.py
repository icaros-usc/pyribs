"""Extension for minifying content.

Requires minify-html to be installed: https://github.com/wilsonzlin/minify-html
"""

from pathlib import Path

import minify_html
from sphinx.util import logging

logger = logging.getLogger(__name__)


def minify_html_files(app, exception):
    """Minifies all HTML files after build is done."""
    if exception is not None:
        return  # Build failed, skip.

    if not app.config.minify_files:
        logger.info("Skipping minifying")
        return

    outdir = Path(app.outdir)
    html_files = list(outdir.rglob("*.html"))
    logger.info(f"Minifying {len(html_files)} HTML files...")

    for path in html_files:
        try:
            text = path.read_text(encoding="utf-8")
            minified = minify_html.minify(text, minify_js=True, minify_css=True)
            path.write_text(minified, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to minify {path}: {e}")


def setup(app):
    """Installs the extension."""
    app.add_config_value("minify_files", False, "html")
    app.connect("build-finished", minify_html_files)
