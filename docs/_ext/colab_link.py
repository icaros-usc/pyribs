"""Extension to add Colab link to notebooks."""

import re
from pathlib import Path

from sphinx.util import logging

logger = logging.getLogger(__name__)


def add_colab_link(
    app,
    pagename,
    templatename,  # pylint: disable = unused-argument
    context,
    doctree,  # pylint: disable = unused-argument
):
    """Inserts a Colab button after the first h1 element in each Jupyter notebook."""
    src_path = Path(app.srcdir) / (pagename + ".ipynb")
    if not src_path.exists():
        return

    base_url = app.config.colab_base_url
    if not base_url:
        logger.warning("colab_base_url is not set. Cannot generate Colab links.")
        return

    notebook_url = f"{base_url}/{pagename + '.ipynb'}"
    colab_url = f"https://colab.research.google.com/github/{notebook_url}"
    context["open_in_colab_url"] = colab_url

    if app.config.colab_auto_insert and "body" in context:
        button_html = f"""\
<p>
  <a href="{colab_url}" target="_blank" rel="noopener noreferrer">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" f'alt="Open in Colab"/>
  </a>
</p>
"""

        body_html = context["body"]

        # Hack the style of the first h1 to make things appear more nicely. `1`
        # indicates to only replace the first occurrence.
        body_html = body_html.replace("<h1 ", '<h1 style="margin-bottom: 0.6em"', 1)

        # Insert right after the first <h1 ...>...</h1>
        def insert_after_first_h1(match):
            return match.group(0) + button_html

        # The (?is) flags make . match newlines and ignore case for <h1>
        body_html, count = re.subn(
            r"(?is)(<h1\b.*?>.*?</h1>)", insert_after_first_h1, body_html, count=1
        )

        if count == 0:
            logger.warning(f"No <h1> found in {pagename}, inserting badge at top.")
            body_html = button_html + body_html

        context["body"] = body_html


def setup(app):
    """Installs the extension."""
    app.add_config_value("colab_base_url", None, "html")
    app.add_config_value("colab_auto_insert", False, "html")
    app.connect("html-page-context", add_colab_link)
