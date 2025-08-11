"""Extension for adding links to GitHub issues and PRs.

Two new Sphinx roles are provided -- :issue: and :pr:. For instance, you can use
:issue:`123` and :pr:`123` to refer to issue 123 and PR 123, respectively.

The link to the GitHub repo must be configured with the github_repo_url config
in conf.py.

References:
- https://doughellmann.com/posts/defining-custom-roles-in-sphinx/
- https://www.sphinx-doc.org/en/master/development/tutorials/helloworld.html
"""

from docutils import nodes, utils
from docutils.parsers.rst.roles import set_classes


def make_link_node(rawtext, app, link_type, slug, options):
    """Creates a link to a GitHub resource.

    Args:
        rawtext: Text being replaced with link node.
        app: Sphinx application context
        link_type: Link type ("issues", "pull")
        slug: ID of the thing to link to
        options: Options dictionary passed to role func.
    """
    try:
        base = app.config.github_repo_url
        if not base:
            raise AttributeError
    except AttributeError as e:
        raise ValueError("github_repo_url configuration value is not set") from e

    slash = "/" if base[-1] != "/" else ""
    ref = f"{base}{slash}{link_type}/{slug}/"
    set_classes(options)
    node = nodes.reference(rawtext, f"#{utils.unescape(slug)}", refuri=ref, **options)
    return node


def github_link_role(
    name,
    rawtext,
    text,
    lineno,
    inliner,
    options=None,
    content=(),  # pylint: disable = unused-argument
):
    """Link to a GitHub issue or pull request.

    Returns 2-part tuple containing list of nodes to insert into the document
    and a list of system messages. Both are allowed to be empty.

    Args:
        name: The role name used in the document.
        rawtext: The entire markup snippet, with role.
        text: The text marked with the role.
        lineno: The line number where rawtext appears in the input.
        inliner: The inliner instance that called us.
        options: Directive options for customization.
        content: The directive content for customization.
    """
    options = options or {}

    try:
        issue_num = int(text)
        if issue_num <= 0:
            raise ValueError
    except ValueError:
        msg = inliner.reporter.error(
            "GitHub issue number must be a number greater than or equal to 1; "
            f"'{text}' is invalid.",
            line=lineno,
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    app = inliner.document.settings.env.app
    link_type = {
        # GitHub issue paths use "issues", like
        # https://github.com/icaros-usc/pyribs/issues/570/
        "issue": "issues",
        # GitHub PR paths use "pull", like
        # https://github.com/icaros-usc/pyribs/pull/575/
        "pr": "pull",
    }[name]
    node = make_link_node(rawtext, app, link_type, str(issue_num), options)
    return [node], []


def setup(app):
    """Installs the extension."""
    app.add_role("issue", github_link_role)
    app.add_role("pr", github_link_role)
    app.add_config_value("github_repo_url", None, "env")
