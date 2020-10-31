# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in many ways.

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/icaros-usc/ribs/issues>

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
<https://github.com/icaros-usc/ribs/issues>

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are
  welcome :)

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

3. Install your local copy into a virtualenv. Assuming you have
   virtualenvwrapper installed, this is how you set up your fork for local
   development. You can also use a Conda environment if you would like.

   ```bash
   mkvirtualenv ribs
   cd ribs/
   pip install -e .
   pip install -r requirements_dev.txt
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
   pytest
   python setup.py test # ^ same as above
   make test # ^ also same as above
   tox  # Don't worry if this does not run; we will run it in CI/CD
   ```

   If you wish to run the tests without benchmarks (which can take a while),
   run:

   ```bash
   pytest -c pytest_no_benchmark.ini
   make test-only # ^ same as above, but shorter
   ```

   To get pylint, pylint, and tox, pip install them into your virtualenv. They
   should already be in `requirements_dev.txt`, however.

6. Commit your changes and push your branch to GitHub::

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
   display results of tests at the bottom of the PR.

### Style

Code should follow the
[Google Style Guide](https://google.github.io/styleguide/pyguide.html) and be
auto-formatted using [YAPF](https://github.com/google/yapf).

## Tips

### Running a Subset of Tests

To run a subset of tests:

```bash
pytest tests.test_ribs
```

### Previewing Documentation

Preview documentation with:

```bash
make servedocs
```

This will display a link at which to view the docs, most likely
<http://localhost:8000>. Visit this link to view the docs; the new pages will
reload automatically.

## Deploying

A reminder for the maintainers on how to deploy. Make sure all your changes are
committed (including an entry in HISTORY.rst). Then run::

```bash
bump2version patch # possible: major / minor / patch
git push
git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
