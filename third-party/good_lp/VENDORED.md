# Vendored dependency

| | |
|---|---|
| Source | https://github.com/hlvlad/good_lp |
| Revision | `0b5a4e97b645f390cf80ef3c68b23fd5cf681491` |
| Upstream | https://github.com/rust-or/good_lp |
| Version | 1.14.0 |
| Licence | MIT -- see `LICENSE` in this directory |

A fork of `good_lp` adding the `gurobi` backend that links `grb` directly.
`augmecon-rs` and `psbox` depend on it by path, so a build of this deposit
fetches nothing from a personal repository.
