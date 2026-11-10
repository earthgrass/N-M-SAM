# Bundle verification portability diagnostic

The v3 tarball itself matched the registered SHA256. After extraction, Python `splitlines()` plus per-file SHA256 verified all 77 entries with zero failures.

The generated `SHA256SUMS.txt` uses CRLF line endings. GNU `sha256sum -c` retains the carriage return at the end of each filename and reports every path as missing. The Bundle must not be silently repacked because Phase 4 authorized one exact tarball hash. For a later bundle revision, generate the manifest with explicit LF newlines and test it on Linux before freezing the archive hash.
