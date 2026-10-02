
# Contributing to Slenderpy

Thank you for your interest in Slenderpy.
If you consider contributing to this repository, please read the relevant sections below.

## Questions and issues

The best place to ask questions and report issues is the [Issue](https://github.com/rte-france/pallas-slenderpy/issues/) section of the [Github repository](https://github.com/rte-france/pallas-slenderpy).
You can use the search function of Github to look for previous similar questions.
For sensitive issues or to request private paid support, you can send a message to <opensource@mews-labs.com>.

## Contributing to the code

Before starting to edit the code to make a change, it is often advisable to open a discussion thread as mentionned above to discuss with the maintainers how it fits in the scope and aims of the project.

When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project licence.

You will then need to submit a pull request (PR). To do so, you will need to:
- Create a fork of Slenderpy on Github (that is a copy on which you have full write access).
- Clone your fork locally on your computer (or add your fork as a new remote to a local git repository).
- On your local git repository, create a new branch dedicated to the bug fix or feature that you want to contribute.
- Create as many commits as you'd like in this branch.
- Push the branch to your fork on Github.
- On Github's web interface, create a new pull request in the main Slenderpy repository at https://github.com/rte-france/pallas-slenderpy, requesting to merge the branch on your fork into the main branch of the original Slenderpy repository.
- Until the pull request is accepted, the maintainers can edit the branch on your fork with you.
- Once the pull request is merged, the branch can be deleted or forgotten.

For simple modifications (e.g. typos) most of the process above can be done automatically by Github by using its edition functionality (the pencil icon on the top left of Github's file viewer).

The code follows the PEP8 guidelines for code style. Indentation is done with four spaces. Try to avoid trailing whitespaces whenever possible.

## Tests and coverage

```shell
uv sync --all-extras
uv run pytest test --cov=slenderpy --cov-report=term --cov-report=html
```

The HTML report is written to `htmlcov/index.html`. The report of the `main`
branch is published with the documentation, at
https://rte-france.github.io/pallas-slenderpy/coverage/; until the docs
workflow has run once, the coverage badge of the README shows as broken.

## Documentation

The documentation is a [Quarto](https://quarto.org/) website in `doc/`; the
API reference is generated from the docstrings (numpy style) by
[quartodoc](https://machow.github.io/quartodoc/). Quarto itself is installed
by the `docs` extra.

```shell
uv sync --all-extras
cd doc
uv run quartodoc build
uv run quartodoc interlinks
uv run quarto render
```

The site is written to `doc/_site/index.html`; `uv run quarto preview`
serves it with live reload.

On Windows, installing `quarto-cli` can fail on the 260-character path
limit. Either enable long paths, or point the uv cache and the temporary
directory to short paths before `uv sync` (PowerShell):

```powershell
$env:UV_CACHE_DIR = "C:/q/c"; $env:TMP = "C:/q/t"; $env:TEMP = "C:/q/t"
```

Pages with executed code (examples, getting started) are frozen: their
results are stored in `doc/_freeze/`, committed, and reused by later builds,
including the docs workflow, which never runs them. When a page or its
example script changes, re-render that page locally (a single page is always
executed), for example `uv run quarto render examples/cable_dynamic.qmd`,
and commit the updated `doc/_freeze/`. A project render only checks the page
source: a change to an example script alone is not detected. The
stockbridge example takes over an hour to run.

A new public module must be added to the `quartodoc` sections of
`doc/_quarto.yml`; `test/test_documentation.py` fails otherwise.
