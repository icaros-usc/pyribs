# Contributing

Contributions are welcome, and they are greatly appreciated. Every little bit
helps, and credit will always be given.

## Types of Contributions

- **Report Bugs:** Refer to the
  [Issue Tracker](https://github.com/icaros-usc/pyribs/issues). Please include
  details such as operating system, Python version, and ribs version, as well as
  detailed steps to reproduce the bug.
- **Fix Bugs:** Look through the Issue Tracker for bugs. Anything tagged with
  "bug" and "help wanted" is open to whoever wants to implement it.
- **Propose features:** To request new features in pyribs, submit a Feature
  Request on the Issue Tracker. In the request, please:
  - Explain in detail how the feature would work.
  - Keep the scope as narrow as possible, to make the features easier to
    implement.
- **Implement Features:** Look through the Issue Tracker for features. Anything
  tagged with "enhancement" and "help wanted" is open to whoever wants to
  implement it.
- **Write Documentation:** pyribs could always use more documentation, whether
  as part of the official pyribs docs, in docstrings, or even on the web in blog
  posts, articles, and such. For the website, refer to the
  [website repo](https://github.com/icaros-usc/pyribs.org).
- **Submit Feedback:** The best way to send feedback is to file an issue on the
  [Issue Tracker](https://github.com/icaros-usc/pyribs/issues).

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

1. Install the local copy and dev requirements into an environment. For
   instance, with Conda, the following creates an environment at `./env`.

   ```bash
   cd pyribs
   conda create --prefix ./env python=3.7 # 3.7 is the minimum version pyribs supports.
   conda activate ./env
   pip install -e .[dev]
   ```

1. Create a branch for local development:

   ```bash
   git checkout -b name-of-bugfix-or-feature
   ```

   Now make the appropriate changes locally.

   - Please follow the
     [Google Style Guide](https://google.github.io/styleguide/pyguide.html)
     (particularly when writing docstrings).
   - Make sure to auto-format the code using YAPF. We highly recommend
     installing an editor plugin that auto-formats on save, but YAPF also runs
     on the command line:

     ```bash
     yapf -i FILES
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

   Finally, to lint the code:

   ```bash
   pylint ribs tests benchmarks examples
   make lint # ^ same as above
   ```

   To get pytest and pylint, pip install them into the environment. However,
   they should already install with `pip install -e .[dev]`.

1. Add your change to the changelog for the current version in `HISTORY.md`.

1. Commit the changes and push the branch to GitHub:

   ```bash
   git add .
   git commit -m "Detailed description of changes."
   git push origin name-of-bugfix-or-feature
   ```

1. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before submitting a pull request, check that it meets these guidelines:

1. Style: Code should follow the
   [Google Style Guide](https://google.github.io/styleguide/pyguide.html) and be
   auto-formatted with [YAPF](https://github.com/google/yapf).
1. The pull request should include tests.
1. If the pull request adds functionality, corresponding docstrings and other
   documentation should be updated.
1. The pull request should work for Python 3.7 and higher. GitHub Actions will
   display test results at the bottom of the pull request page. Check there for
   test results.

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

1. Write the notebook and save it under `tutorials/`.
1. Use cell magic (e.g. `%pip install ribs`) to install pyribs and other
   dependencies.
   - Installation cells tend to produce a lot of output. Make sure to clear this
     output in Jupyter lab so that it does not clutter the documentation.
1. Before the main loop of the QD algorithm, include a line like
   `total_itrs = 500` (any other integer will work). This line will be replaced
   during testing (see `tests/tutorials.sh`) in order to test that the notebook
   runs end-to-end. By default, the tests run the notebook with
   `total_itrs = 5`. If this tutorial needs more (or less), modify the
   switch-case statement in `tests/tutorials.sh`.
1. Make sure that the only level 1 heading (e.g. `# Awesome Tutorial`) is the
   title at the top of the notebook. Subsequent titles should be level 2 (e.g.
   `## Level 2 Heading`) or higher.
1. If linking to the pyribs documentation, make sure to link to pages in the
   `latest` version on ReadTheDocs, i.e. your links should start with
   `https://docs.pyribs.org/en/latest/`
1. Add an entry into the toctree in `docs/tutorials.md` and add it to one of the
   lists of tutorials.
1. Check that the tutorial shows up on the Tutorials page when serving the docs.
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
Also, prefer to link to the paper's website, rather than just the PDF. This is
particularly relevant when linking to arXiv papers.

### Deploying

1. Create a PR into master after doing the following:
   1. Switch tutorial links from latest to stable with:
      ```bash
      make tutorial_links
      ```
      See [#300](https://github.com/icaros-usc/pyribs/pull/300) for why we do
      this.
   1. Update the version with `bump2version` by running the following for minor
      versions:
      ```bash
      bump2version minor
      ```
      or the following for patch versions:
      ```bash
      bump2version patch
      ```
   1. Add all necessary info on the version to `HISTORY.md`.
1. (Optional) Once the PR has passed CI/CD and been squashed-and-merged into
   master, check out the squash commit and locally run `make release-test`. This
   uploads the code to TestPyPI to check that the deployment works. If this
   fails, make fixes as appropriate.
1. Once the PR in step 1 and any changes in step 2 have passed CI/CD and been
   squashed-and-merged into master, locally tag the master branch with a tag
   like `v0.2.1`, e.g.
   ```bash
   git tag v0.2.1 HEAD
   ```
1. Now push the tag with
   ```bash
   git push --tags
   ```
1. Check that the version was deployed to PyPI. If it failed, delete the tag,
   make appropriate fixes, and repeat steps 2 and 3.
1. Write up the release on GitHub, and attach it to the tag.
1. Submit another PR which reverts the changes to the tutorial links.
   Specifically, while on master, make sure your workspace is clean, then revert
   the changes with:
   ```bash
   git checkout HEAD~ tutorials/
   ```
   And commit the result.

Our deployment process may change in the future as pyribs becomes more complex.
