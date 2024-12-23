# Binius Models

Binius Models is a collection of Python models of cryptographic algorithms and protocols. They are intended for purposes of readability and communication rather than high-performance.

For now, the goal is to present a minimal implementation of FRI-Binius, a multilinear polynomial commitment scheme over binary tower fields, developed in [[DP24](https://eprint.iacr.org/2024/504.pdf)]. Also included is an implementation of the Vision Mark-32 hash function, as developed in [[AMPŠ24](https://eprint.iacr.org/2024/633)].

## Dependencies

This project uses

- [pyenv](https://github.com/pyenv/pyenv) to manage Python versions. [Installation](https://github.com/pyenv/pyenv#installation).
- [poetry](https://python-poetry.org/) to manage the Python environment and dependencies. [Installation](https://python-poetry.org/docs/#installation).
- [pre-commit](https://pre-commit.com/) to automatically deal with simple code issues (e.g., formatting and static analysis). [Installation](https://pre-commit.com/#install).
- [SageMath](https://www.sagemath.org/) to validate Python implementations against.

### SageMath

SageMath is a popular open-source language for math, based on Python.
We use Sage to generate data for tests, specifically `vision`. However, Sage is not required to run any of the tests.

We require Sage version 10, which unfortunately is not currently distributed by standard Linux package managers like `apt` on Ubuntu.
The recommended workaround is to run Sage in a Docker container using the standard [`sagemath/sagemath`](https://hub.docker.com/r/sagemath/sagemath) Docker image.

In this case, [install Docker](https://docs.docker.com/engine/install/) and test it with the following:

```bash
$ docker run --rm sagemath/sagemath sage --version
SageMath version 10.0, Release Date: 2023-05-20
```

Then set the environment variable `export SAGE_IN_DOCKER=1` before running tests.

## Development

With pyenv correctly installed and configured, install the currently used version of Python:

```bash
$ pyenv install
```

Check that the `pyenv` local version is correct:

```bash
$ pyenv version
3.13.0 (set by <CWD>/.python-version)
```

Set up the poetry environment using `pyenv`:

```bash
$ poetry env use $(pyenv which python)
```

Install dependencies with

```bash
$ poetry install
```

Run the type checker with

```bash
$ poetry run mypy
```

Run tests with

```bash
$ poetry run pytest -m "not slow"
```

Install and configure pre-commit

`pre-commit` can be installed locally with:
```
pip install pre-commit
```

The `pre-commit` hooks are configured in the file `.pre-commit-config.yaml` at the root directory. The hooks can be installed with the following command.

```
pre-commit install
```

The `pre-commit` hooks, once installed, will be triggered automatically when commit is made. However, it is possible to run it manually with the following command.
```
pre-commit run --all-files
```

To update hooks from time to time run
```
pre-commit autoupdate
```

## License

   Copyright 2023-2024 Irreducible Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
