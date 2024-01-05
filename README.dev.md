# `scikit-talk` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

```shell
# Create a virtual environment, e.g. with
python3 -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python3 -m pip install --upgrade pip setuptools

# (from the project root directory)
# install scikit-talk as an editable package
python3 -m pip install --no-cache-dir --editable .
# install development dependencies
python3 -m pip install --no-cache-dir --editable .[dev]
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

The second is to use `tox`, which can be installed separately (e.g. with `pip install tox`), i.e. not necessarily inside the virtual environment you use for installing `scikit-talk`, but then builds the necessary virtual environments itself by simply running:

```shell
tox
```

Testing with `tox` allows for keeping the testing environment separate from your development environment.
The development environment will typically accumulate (old) packages during development that interfere with testing; this problem is avoided by testing with `tox`.

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

## Running linters locally

For linting we will use [prospector](https://pypi.org/project/prospector/) and to sort imports we will use
[isort](https://pycqa.github.io/isort/). Running the linters requires an activated virtual environment with the
development tools installed.

```shell
# linter
prospector

# recursively check import style for the scikit-talk module only
isort --check-only scikit-talk

# recursively check import style for the scikit-talk module only and show
# any proposed changes as a diff
isort --check-only --diff scikit-talk

# recursively fix import style for the scikit-talk module only
isort scikit-talk
```

To fix readability of your code style you can use [yapf](https://github.com/google/yapf).

You can enable automatic linting with `prospector` and `isort` on commit by enabling the git hook from `.githooks/pre-commit`, like so:

```shell
git config --local core.hooksPath .githooks
```

## Generating the API docs

```shell
cd docs
make html
```

The documentation will be in `docs/_build/html`

If you do not have `make` use

```shell
sphinx-build -b html docs docs/_build/html
```

To find undocumented Python objects run

```shell
cd docs
make coverage
cat _build/coverage/python.txt
```

To [test snippets](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) in documentation run

```shell
cd docs
make doctest
```

## Versioning

Bumping the version across all files is done with [bumpversion](https://github.com/c4urself/bump2version), e.g.

```shell
bumpversion major
bumpversion minor
bumpversion patch
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on GitHub
1. verification

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
1. Verify that the information in `CITATION.cff` is correct
1. Make sure the [version has been updated](#versioning).
1. Run the unit tests with `pytest -v`

### (2/3) Release on GitHub

Releases start on GitHub.
Open [releases](https://github.com/elpaco-escience/scikit-talk/releases/new) and draft a new release.
Copy the changelog for this version into the description.
Tag the release according to [semantic versioning guidelines](https://semver.org/), preceded with a `v` (e.g.: v1.0.0).
The release title is the tag and the release date together (e.g.: v1.0.0 (2019-07-25)).

The Zenodo integration will take care of updating the Zenodo record with a new release.
Releasing on GitHub will also automatically trigger the publication of the release to PyPI, through a Github Action.
To verify that everything works as expected, it is recommended to first publish a release candidate (see [below](#release-candidates)).

> [!NOTE]
>
> #### Release candidates
>
> When releasing a release candidate on Github, tick the pre-release box, and amend the version tag with `-rc` and the candidate number (e.g. v1.0.0-rc1).
> Ensure the release candidate version is accurate in `CITATION.cff`.
> Versions with "rc" (release candidate) in their version tag will only be published to [test.PyPI](https://test.pypi.org/project/test-scikit-talk/).
> Other version tags will trigger a [PyPI release](https://pypi.org/project/scikit-talk/).
> Inspect `.github/workflows/publish-pypi.yml` for more information.

### (3/3) Verification

1. Verify the publication to [testPyPI](https://test.pypi.org/project/test-scikit-talk/) or [PyPI](https://pypi.org/project/scikit-talk/) (depending on the version tag).
1. Confirm that the released package can be installed
    - from PyPI with `pip install scikit-talk`
    - from test.PyPI with `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/test-scikit-talk`
1. The release should have triggered a [Zenodo upload](https://zenodo.org/). Confirm that the Zenodo record has been updated.
