//! The local path a file name or a `file:` URL names, as every statement and procedure that
//! reads a file takes it (`LOAD CSV`, `LOAD DATA`, the load procedures): RFC 8089.
//!
//! - A name without the `file:` scheme is a path, used as given (relative to the working
//!   directory, or absolute).
//! - `file:///abs/path` and `file://localhost/abs/path` name `/abs/path`; on Windows
//!   `file:///C:/data/x.csv` names `C:/data/x.csv`. `file:/abs/path` (no authority) names the
//!   same.
//! - Any other host is refused: only local files are read.
//! - `%XX` escapes in the path are decoded (`%20` is a space).

use std::path::PathBuf;

/// A `file:` URL that names no local file.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FileUrlError {
    /// The URL names a host other than this machine.
    #[error(
        "the file URL '{url}' names the host '{host}': only local files are read \
         (file:///path or file://localhost/path)"
    )]
    RemoteHost {
        /// The URL as given.
        url: String,
        /// The host it names.
        host: String,
    },
    /// The path holds a `%` escape that is not two hex digits, or escapes that are not UTF-8.
    #[error("the file URL '{url}' holds an escape that is not a UTF-8 character")]
    InvalidEscape {
        /// The URL as given.
        url: String,
    },
}

/// The local path `name` names: a path as given, or the path of a `file:` URL (RFC 8089, see
/// the module docs).
///
/// # Errors
///
/// [`FileUrlError::RemoteHost`] for a URL of another host, [`FileUrlError::InvalidEscape`] for
/// a path whose escapes do not decode.
pub fn local_path(name: &str) -> Result<PathBuf, FileUrlError> {
    let Some(rest) = strip_scheme(name) else {
        return Ok(PathBuf::from(name));
    };
    let path = match rest.strip_prefix("//") {
        Some(authority_and_path) => {
            let (host, path) = authority_and_path
                .find('/')
                .map_or((authority_and_path, ""), |slash| {
                    authority_and_path.split_at(slash)
                });
            if !host.is_empty() && !host.eq_ignore_ascii_case("localhost") {
                return Err(FileUrlError::RemoteHost {
                    url: name.to_string(),
                    host: host.to_string(),
                });
            }
            path
        }
        None => rest,
    };
    let decoded = decode(path).ok_or_else(|| FileUrlError::InvalidEscape {
        url: name.to_string(),
    })?;
    Ok(PathBuf::from(without_drive_slash(decoded)))
}

/// What follows `file:` (the scheme matched without regard to case), `None` for a plain path.
fn strip_scheme(name: &str) -> Option<&str> {
    let scheme = name.get(..5)?;
    scheme.eq_ignore_ascii_case("file:").then(|| &name[5..])
}

/// `path` with its `%XX` escapes decoded, `None` when one is malformed or the bytes are not
/// UTF-8.
fn decode(path: &str) -> Option<String> {
    if !path.contains('%') {
        return Some(path.to_string());
    }
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'%' {
            let hex = path.get(at + 1..at + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            at += 3;
        } else {
            out.push(bytes[at]);
            at += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// On Windows, `/C:/x` (a URL path with a drive) as the path `C:/x`; elsewhere unchanged.
fn without_drive_slash(path: String) -> String {
    if cfg!(windows) && starts_with_drive_slash(&path) {
        path[1..].to_string()
    } else {
        path
    }
}

/// Whether `path` reads `/X:` or `/X|` for a drive letter `X`.
fn starts_with_drive_slash(path: &str) -> bool {
    let bytes = path.as_bytes();
    bytes.len() >= 3
        && bytes[0] == b'/'
        && bytes[1].is_ascii_alphabetic()
        && (bytes[2] == b':' || bytes[2] == b'|')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> String {
        local_path(name).unwrap().to_string_lossy().into_owned()
    }

    #[test]
    fn a_plain_path_is_used_as_given() {
        assert_eq!(path("data/people.csv"), "data/people.csv");
        assert_eq!(path("/tmp/people.csv"), "/tmp/people.csv");
        assert_eq!(path("C:/data/people.csv"), "C:/data/people.csv");
    }

    #[test]
    fn a_file_url_names_its_absolute_path() {
        assert_eq!(path("file:///tmp/people.csv"), "/tmp/people.csv");
        assert_eq!(path("file://localhost/tmp/people.csv"), "/tmp/people.csv");
        assert_eq!(path("FILE://LocalHost/tmp/people.csv"), "/tmp/people.csv");
        assert_eq!(path("file:/tmp/people.csv"), "/tmp/people.csv");
        assert_eq!(path("file:///data/people.csv"), "/data/people.csv");
    }

    #[test]
    fn escapes_are_decoded() {
        assert_eq!(
            path("file:///tmp/Amsterdam%20Zuid/caf%C3%A9.csv"),
            "/tmp/Amsterdam Zuid/caf\u{e9}.csv"
        );
        assert!(matches!(
            local_path("file:///tmp/a%2.csv"),
            Err(FileUrlError::InvalidEscape { .. })
        ));
        assert!(matches!(
            local_path("file:///tmp/a%FF.csv"),
            Err(FileUrlError::InvalidEscape { .. })
        ));
    }

    #[test]
    fn another_host_is_refused() {
        let refused = local_path("file://server/share/people.csv").unwrap_err();
        assert_eq!(
            refused,
            FileUrlError::RemoteHost {
                url: "file://server/share/people.csv".into(),
                host: "server".into()
            }
        );
        assert!(refused.to_string().contains("only local files are read"));
        assert!(matches!(
            local_path("file://data.csv"),
            Err(FileUrlError::RemoteHost { .. })
        ));
    }

    #[test]
    fn a_windows_drive_keeps_its_letter_on_windows() {
        let expected = if cfg!(windows) {
            "C:/data/people.csv"
        } else {
            "/C:/data/people.csv"
        };
        assert_eq!(path("file:///C:/data/people.csv"), expected);
    }
}
