# Quick and dirty CGNS API for Rust

This is an unmaintained idiomatic wrapper for libcgns.

## Using

If needed add this to `.cargo/config.tom`:

```toml
[env]
LIBRARY_PATH="/path/to/libcgns/lib"
CGNS_INCLUDE_DIR="/path/to/libcgns/include"
```
or

```toml
[env]
CGNS_DIR="/path/to/libcgns
```

and

```
cargo add --git https://github.com/tucanos/qd-cgns-rs.git
```
