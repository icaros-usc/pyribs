# Contributing

Contributions are welcome and are greatly appreciated! Every little bit helps.
Contributions include reporting/fixing bugs, proposing/implementing features
(see our [Issue Tracker](https://github.com/icaros-usc/pyribs/issues)), writing
documentation in the codebase or on our
[website repo](https://github.com/icaros-usc/pyribs.org), and submitting
feedback.

## Developing pyribs

Ready to contribute? Here's how to set up pyribs for local development.

1. [Fork](https://github.com/icaros-usc/pyribs/fork) the pyribs repo on GitHub.
1. Clone the fork locally:

   ```bash
   # With SSH:
   git clone git@github.com:USERNAME/pyribs.git

   # Without SSH:
   git clone https://github.com/USERNAME/pyribs.git
   ```

1. Create a branch for local development:

   ```bash
   git checkout -b name-of-bugfix-or-feature
   ```

1. Install the local copy and dev requirements into a virtual environment. For
   instance, with Conda, the following creates an environment at `./env`.

   ```bash
   cd pyribs
   conda create --prefix ./env python=3.9  # 3.9 is the minimum version pyribs supports.
   conda activate ./env
   pip install -e .[all,dev]  # `all` contains dependencies for all extras of pyribs.
                              # `dev` contains development dependencies.
   ```

1. We roughly follow the
   [Google Style Guide](https://google.github.io/styleguide/pyguide.html) in our
   codebase and use isort, ruff, and pylint to enforce code format and style. To
   automatically check for formatting and style every time you commit, we use
   [pre-commit](https://pre-commit.com). Pre-commit should have already been
   installed with `.[dev]` above. To set it up, run:

   ```bash
   pre-commit install
   ```

1. Now make the appropriate changes locally. If relevant, make sure to write
   tests for your code in the `tests/` folder.

1. Auto-format and lint your code using [isort](https://pycqa.github.io/isort/),
   [ruff](https://docs.astral.sh/ruff/formatter/), and
   [pylint](https://www.pylint.org/). Note that pre-commit will automatically
   run these whenever you commit your code; you can also run them with
   `pre-commit run`. You can also run these commands on the command line:

   ```bash
   isort FILES
   ruff format FILES
   pylint FILES
   ```

1. After making changes, check that the changes pass the tests:

   ```bash
   pytest tests/
   make test # ^ same as above
   ```

   And to run the benchmarks:

   ```bash
   pytest -c pytest_benchmark.ini
   make benchmark # ^ same as above
   ```

1. Add your change to the changelog for the current version in `HISTORY.md`.

1. Commit the changes and push the branch to GitHub:

   ```bash
   git add .
   git commit -m "Detailed description of changes."
   git push origin name-of-bugfix-or-feature
   ```

1. Submit a pull request through the GitHub web interface.

## Instructions

### Running a Subset of Tests

To run a subset of tests, use `pytest` with the directory name, such as:

```bash
pytest tests/core/archives
```

### Documentation

Documentation is primarily written in Markdown, as we use the
[MyST](https://myst-parser.readthedocs.io/en/latest/) Sphinx plugin.

To preview documentation, use:

```bash
make servedocs
```

This will open up a browser window and automatically reload as changes are made
to the docs.

### Adding a Tutorial

Tutorials are created in Jupyter notebooks that are stored under `tutorials/` in
the repo. To create a tutorial:

1. Write the notebook and save it under `tutorials/`. Notebooks may also be
   saved in a subdirectory of `tutorials/`, e.g., `tutorials/features`.
1. Use cell magic (e.g. `%pip install ribs`) to install pyribs and other
   dependencies.
   - Installation cells tend to produce a lot of output. Make sure to clear this
     output in Jupyter lab so that it does not clutter the documentation.
   - Choose your ribs extra carefully -- in particular, if you use
     `ribs.visualize` in the tutorial, make sure to use
     `%pip install ribs[visualize]`.
1. Before the main loop of the QD algorithm, include a line like
   `total_itrs = 500` (any other integer will work). This line will be replaced
   during testing (see `tests/tutorials.sh`) in order to test that the notebook
   runs end-to-end. By default, the tests run the notebook with
   `total_itrs = 5`. If this tutorial needs more (or less), modify the
   switch-case statement in `tests/tutorials.sh`. The name `TOTAL_ITRS` can also
   be used.
1. Make sure that the only level 1 heading (e.g. `# Awesome Tutorial`) is the
   title at the top of the notebook. Subsequent titles should be level 2 (e.g.
   `## Level 2 Heading`) or higher.
1. If linking to the pyribs documentation, make sure to link to pages in the
   `latest` version on ReadTheDocs, i.e. your links should start with
   `https://docs.pyribs.org/en/latest/`. Note that we do not use Sphinx autodoc
   (e.g., `:class:`) in the tutorials because such links do not work outside the
   pyribs website (e.g., on Google Colab).
1. Add an entry into the toctree in `docs/tutorials.md` and add it to one of the
   lists of tutorials.
1. Check that the tutorial shows up on the Tutorials page when serving the docs.
1. If the tutorial should be excluded from testing (e.g., because it takes too
   long to run), add it to the list of excluded tutorials in
   `tests/tutorials_list.sh`.
1. Create a PR into the website repo that adds the tutorial onto the home page,
   specifically
   [this file](https://github.com/icaros-usc/pyribs.org/blob/master/src/index.liquid).
   In the PR, include a square image that represents the tutorial.

### Adding an Example

Examples are created in Python scripts stored under `examples/` in the repo, and
their source is shown in the docs. To create an example:

1. Write the Python script and save it under `examples/`.
1. Add any dependencies at the top of the script with a `pip install` command
   (see existing examples for a sample of how to do this).
1. Add a shell command to `tests/examples.sh` that calls the script with
   parameters that will make it run as quickly as possible. This helps us ensure
   that the script has basic correctness. Also call the `install_deps` function
   on the script file before running the script.
1. Add a Markdown file in the `docs/examples` directory with the same name as
   the Python file -- if the example is `examples/foobar.py`, the Markdown file
   will be `docs/examples/foobar.md`.
1. Add a title to the Markdown file, such as:
   ```
   # My Awesome Example
   ```
1. In the markdown file, include the following so that the source code of the
   example is displayed.
   ````
   ```{eval-rst}
   .. literalinclude:: ../../examples/EXAMPLE.py
       :language: python
       :linenos:
   ```
   ````
1. Add any other relevant info to the Markdown file.
1. Add an entry into the toctree and list of examples in `docs/examples.md`.
1. Check that the example shows up on the Examples page when serving the docs.

### Referencing Papers

When referencing papers, refer to them as `Lastname YEAR`, e.g. `Smith 2004`.
Also, prefer to link to the paper's website, rather than just the PDF.

### Deploying

We roughly follow the trunk-based release model described
[here](https://trunkbaseddevelopment.com/branch-for-release/). Namely, we create
release branches for minor versions (e.g., `release/0.6.x`) from our trunk
(`master` branch). For patch releases, e.g., `0.6.2`, we cherry pick bug fixes
from `master` into the corresponding release branch.

#### Minor Versions

1. Check that the latest version of the docs is building on Read the Docs.
1. Create a PR into master after doing the following:
   1. Update the version by editing `__version__` in `ribs/__init__.py`.
   1. Add all necessary info on the version to `HISTORY.md`.
1. Once the PR above has been merged, create a release branch from master called
   `release/0.<NUM>.x`, e.g., `release/0.6.x`.
1. On the release branch, switch tutorial links from latest to stable with:
   ```bash
   make tutorial_links
   ```
   See [#300](https://github.com/icaros-usc/pyribs/pull/300) for why we do this.
1. (Optional) On the release branch, run `make release-test`. This uploads the
   code to TestPyPI to check that the deployment works. If this fails, make
   fixes as appropriate.
1. Locally tag the head of the release branch, e.g.,
   ```bash
   git tag v0.6.0 HEAD
   ```
1. Now push the tag with
   ```bash
   git push --tags
   ```
1. Check that the version was deployed to PyPI. If it failed, delete the tag and
   make appropriate fixes on master. Then, cherry pick the fixes into the
   release branch. Finally, tag the HEAD again and push the tag.
1. Write up the release on GitHub, and attach it to the tag.

#### Patch Versions

1. Check that the latest version of the docs is building on Read the Docs.
1. Create a PR into master after doing the following:
   1. Update the version by editing `__version__` in `ribs/__init__.py`.
   1. Add all necessary info on the version to `HISTORY.md`.
1. Once the PR above has been merged, checkout the release branch for the
   corresponding minor version, e.g., for `0.6.2`, check out `release/0.6.x`.
1. On the release branch, cherry-pick the commit for the PR you just created.
   Also cherry pick any other bug fixes which need to be released in this patch.
1. On the release branch, edit `HISTORY.md` to remove any irrelevant history,
   e.g., if there are upcoming changes that will be included only in the next
   version.
1. If any tutorials were added in this release, run `make tutorial_links` to
   make the links point to the stable version of pyribs.
1. (Optional) On the release branch, run `make release-test`. This uploads the
   code to TestPyPI to check that the deployment works. If this fails, make
   fixes as appropriate.
1. Locally tag the head of the release branch, e.g.,
   ```bash
   git tag v0.6.2 HEAD
   ```
1. Now push the tag with
   ```bash
   git push --tags
   ```
1. Check that the version was deployed to PyPI. If it failed, delete the tag and
   make appropriate fixes on master. Then, cherry pick the fixes into the
   release branch. Finally, tag the HEAD again and push the tag.
1. Write up the release on GitHub, and attach it to the tag.
