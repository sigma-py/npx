# npx

[![PyPi Version](https://img.shields.io/pypi/v/npx.svg?style=flat-square)](https://pypi.org/project/npx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/npx.svg?style=flat-square)](https://pypi.org/project/npx/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/npx.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/npx)
[![PyPi downloads](https://img.shields.io/pypi/dm/npx.svg?style=flat-square)](https://pypistats.org/packages/npx)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/npx/ci?style=flat-square)](https://github.com/nschloe/npx/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/npx.svg?style=flat-square)](https://app.codecov.io/gh/nschloe/npx)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/npx.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/npx)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

[NumPy](https://numpy.org/) is a large library used everywhere in scientific computing.
That's why breaking backwards-compatibility comes at a significant cost and is almost
always avoided, even if the API of some methods is arguably lacking. This package
provides drop-in wrappers "fixing" those.

[scipyx](https://github.com/nschloe/scipyx) does the same for
[SciPy](https://www.scipy.org/).

If you have a fix for a NumPy method that can't go upstream for some reason, feel free
to PR here.

#### `dot`

```python
import npx
import numpy as np

a = np.random.rand(3, 4, 5)
b = np.random.rand(5, 2, 2)

out = npx.dot(a, b)
# out.shape == (3, 4, 2, 2)
```

Forms the dot product between the last axis of `a` and the _first_ axis of `b`.

(Not the second-last axis of `b` as `numpy.dot(a, b)`.)

#### `np.solve`

```python
import npx
import numpy as np

A = np.random.rand(3, 3)
b = np.random.rand(3, 10, 4)

out = npx.solve(A, b)
# out.shape == (3, 10, 4)
```

Solves a linear equation system with a matrix of shape `(n, n)` and an array of shape
`(n, ...)`. The output has the same shape as the second argument.

#### `sum_at`/`add_at`

<!--pytest-codeblocks:skip-->

```python
npx.sum_at(a, idx, minlength=0)
npx.add_at(out, idx, a)
```

Returns an array with entries of `a` summed up at indices `idx` with a minimum length of
`minlength`. `idx` can have any shape as long as it's matching `a`. The output shape is
`(minlength,...)`.

The numpy equivalent `numpy.add.at` is _much_
slower:

<img alt="memory usage" src="https://nschloe.github.io/npx/perf-add-at.svg" width="50%">

Relevant issue reports:

- [ufunc.at (and possibly other methods)
  slow](https://github.com/numpy/numpy/issues/11156)


#### `unique`

```python
import npx

a = [0.1, 0.15, 0.7]
a_unique, inv = npx.unique(a, tol=2.0e-1, return_inverse=True)

assert all(a_unique == [0.1, 0.7])
```

npx's `unique()` works just like NumPy's, except that it provides a parameter
`tol` (default `0.0`) which allows the user to set a tolerance. The real line
is essentially partitioned into bins of size `tol` and at most one
representative of each bin is returned.


#### `unique_rows`
```python
import npx
import numpy as np

a = np.random.randint(0, 5, size=(100, 2))

npx.unique_rows(a, return_inverse=False, return_counts=False)
```

Returns the unique rows of the integer array `a`. The numpy alternative `np.unique(a, axis=0)` is slow.

Relevant issue reports:

- [unique() needlessly slow](https://github.com/numpy/numpy/issues/11136)

#### `isin_rows`

```python
import npx
import numpy as np

a = [[0, 1], [0, 2]]
b = np.random.randint(0, 5, size=(100, 2))

npx.isin_rows(a, b)
```

Returns a boolean array of length `len(a)` specifying if the rows `a[k]` appear in `b`.
Similar to NumPy's own `np.isin` which only works for scalars.

#### `mean`

```python
import npx

a = [1.0, 2.0, 5.0]
npx.mean(a, p=3)
```

Returns the [generalized mean](https://en.wikipedia.org/wiki/Generalized_mean) of a
given list. Handles the cases `+-np.inf` (max/min) and`0` (geometric mean) correctly.
Also does well for large `p`.

Relevant NumPy issues:

- [generalized mean](https://github.com/numpy/numpy/issues/19341)

### License

This software is published under the [BSD-3-Clause
license](https://spdx.org/licenses/BSD-3-Clause.html).
