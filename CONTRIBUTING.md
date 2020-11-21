# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/icaros-usc/ribs/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

pyribs could always use more documentation, whether as part of the official
pyribs docs, in docstrings, or even on the web in blog posts, articles, and
such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/icaros-usc/ribs/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.

## Get Started!

Ready to contribute? Here's how to set up `ribs` for local development.

1. Fork the `ribs` repo on GitHub.
2. Clone your fork locally:

   ```bash
   # If you have SSH set up:
   git clone git@github.com:your_name_here/pyribs.git

   # Or, if you do not have SSH set up:
   git clone https://github.com/your_name_here/pyribs.git
   ```

3. Install the local copy and dev requirements into an environment. For
   instance, with Conda, you could use:

   ```bash
   cd pyribs
   conda create --prefix ./env python=3.6 # 3.6 is the minimum version pyribs supports.
   conda activate ./env
   pip install -e .[all]
   pip install -e .[dev]
   ```

4. Create a branch for local development:

   ```bash
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

   - Make sure to follow the
     [Google Style Guide](https://google.github.io/styleguide/pyguide.html)
     (particularly when writing docstrings).
   - Make sure to auto-format your code using YAPF. We highly recommend
     installing a plugin to your editor that auto-formats on save, but you can
     also run YAPF on the command line:

     ```bash
     yapf -i FILES
     ```

5. When you're done making changes, check that your changes pass pylint and the
   tests, including testing other Python versions with tox:

   ```bash
   pylint ribs tests
   pytest tests/
   make test # ^ same as above
   tox  # Don't worry if this does not run; we will run it in CI/CD
   ```

   If you wish to run the tests without benchmarks (which can take a while),
   run:

   ```bash
   pytest -c pytest_no_benchmark.ini
   make test-only # ^ same as above, but shorter
   ```

   To get pytest, pylint, and tox, pip install them into your virtualenv. They
   should already install with `pip install -e .[dev]`, however.

6. Commit your changes and push your branch to GitHub:

   ```bash
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring, and add the feature to
   the list in README.rst.
3. The pull request should work for Python 3.6, 3.7 and 3.8. GitHub Actions will
   display test results at the bottom of the pull request page; check there to
   see if your code passes all tests.

### Style

Code should follow the
[Google Style Guide](https://google.github.io/styleguide/pyguide.html) and be
auto-formatted using [YAPF](https://github.com/google/yapf).

## Instructions

### Running a Subset of Tests

To run a subset of tests:

```bash
pytest tests/core/archives
```

### Documentation

In addition to reStructuredText, you can write documentation in Markdown, as we
use [MyST](https://myst-parser.readthedocs.io/en/latest/).

To preview documentation, use:

```bash
make servedocs
```

This will open up a window in your browser, and as you make changes to the docs,
the new pages will reload automatically.

#### Adding a Tutorial

Tutorials are created in Jupyter notebooks that are stored under
`examples/tutorials` in the repo. To create a tutorial:

1. Write the notebook and save it under `examples/tutorials`.
1. Add an "Open in Colab badge" by putting the following Markdown somewhere near
   the beginning of your notebook:

   ```bash
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/icaros-usc/pyribs/blob/master/examples/tutorials/NOTEBOOK.ipynb)
   ```

   Make sure to replace `NOTEBOOK` with the name of your notebook, of course.
   The badge should look like this when rendered (clicking on this specific
   badge will take you to more information about Colab and GitHub):

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

1. Add an entry into the toctree and table in `docs/tutorials.md`.
1. Check that the tutorial shows up on the Tutorials page when you serve the docs.

#### Adding an Example

Examples are created in Python files stored under `examples` in the repo, but
their source is shown in the docs. To create an example:

1. Write the Python file and save it under `examples`.
1. Add a Markdown file in the `docs/examples` directory with the same name as
   your Python file -- if your example is `examples/foobar.py`, your Markdown
   file will be `docs/examples/foobar.md`.
1. Add a title to your Markdown file, such as:
   ```
   # My Awesome Example
   ```
1. In the markdown file, include the following so that the source code of the
   example is displayed.
   ````
   ```{eval-rst}
   .. literalinclude:: ../../examples/YOUR_EXAMPLE.py
       :language: python
       :linenos:
   ```
   ````
1. Add whatever other info you would like in the Markdown file.
1. Add an entry into the toctree and table in `docs/examples.md`.
1. Check that the example shows up on the Examples page when you serve the docs.

### Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are
committed (including an entry in HISTORY.rst). Then run:

```bash
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
