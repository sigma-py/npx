# npx

[![PyPi Version](https://img.shields.io/pypi/v/npx.svg?style=flat-square)](https://pypi.org/project/npx)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/npx.svg?style=flat-square)](https://pypi.org/pypi/npx/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/npx.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/npx)
[![PyPi downloads](https://img.shields.io/pypi/dm/npx.svg?style=flat-square)](https://pypistats.org/packages/npx)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/npx/ci?style=flat-square)](https://github.com/nschloe/npx/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/npx.svg?style=flat-square)](https://codecov.io/gh/nschloe/npx)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/npx.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/npx)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

[NumPy](https://numpy.org/) is a huge library of with useful method for scientific
computing. The API of some methods is arguably confusing, but can't be changed without
breaking backwards compatibility. Some methods can be improved, but it's hard to do so
across the board.

npx provides some drop-in functions for those; mostly they are thin wrappers around
native numpy functions.

##### `npx.dot(a, b)`

Forms the dot product between the last axis of `a` and the _first_ axis of `b`.

(Not the second-last axis of `b` as `numpy.dot(a, b)`.)

##### `npx.solve(A, b)`

Solves a linear equation system with a matrix of shape `(n, n)` and an array of shape
`(n, ...)`. The output has the same shape as the second argument. Uses
`numpy.linalg.solve` internally.


##### `npx.add_at(a, idx, minlength: int)`
Return an array with entries of `a` summed up at indices `idx` with a minumum length of
`minlength`. `idx` can have any shape as long as it's matching `a`. The output shape is
`(minlength,...)`.

The numpy equivalent is `numpy.add.at` [which is _much_
slower](https://github.com/numpy/numpy/issues/11156).

### License
npx is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
