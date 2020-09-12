============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

.. contents ::

You can contribute in many ways.

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/icaros-usc/ribs/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

pyribs could always use more documentation, whether as part of the
official pyribs docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/icaros-usc/ribs/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `ribs` for local development.

1. Fork the `ribs` repo on GitHub.
2. Clone your fork locally::

    $ # If you have SSH set up:
    $ git clone git@github.com:your_name_here/pyribs.git
    $
    $ # Or, if you do not have SSH set up:
    $ git clone https://github.com/your_name_here/pyribs.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development. You can also use a Conda environment if you would like.::

    $ mkvirtualenv ribs
    $ cd ribs/
    $ pip install -e .
    $ pip install -r requirements_dev.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

   * Make sure to follow the `Google Style Guide
     <https://google.github.io/styleguide/pyguide.html>`_ (particularly when
     writing docstrings).
   * Make sure to auto-format your code using YAPF. We highly recommend
     installing a plugin to your editor that auto-formats on save, but you can
     also run YAPF on the command line: ::

       yapf -i FILES

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 ribs tests
    $ python setup.py test or pytest
    $ tox  # Don't worry if this does not run; we will run it in CI/CD

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.6, 3.7 and 3.8. Check
   https://travis-ci.com/icaros-usc/ribs/pull_requests and make sure that the
   tests pass for all supported Python versions.

Style
~~~~~

Code should follow the `Google Style Guide
<https://google.github.io/styleguide/pyguide.html>`_ and be auto-formatted using
`YAPF <https://github.com/google/yapf>`_.

Tips
----

Running a Subset of Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

To run a subset of tests::

$ pytest tests.test_ribs

Previewing Individual reST files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To preview individual reST files outside of the documentation (such as
CONTRIBUTING.rst and README.rst), install `restview
<https://pypi.org/project/restview/>`_ and run::

  restview FILE

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.
