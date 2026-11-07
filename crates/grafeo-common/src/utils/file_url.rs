//! The local path a file name or a `file:` URL names, as every statement and procedure that
//! reads a file takes it (`LOAD CSV`, `LOAD DATA`, the load procedures): RFC 8089.
//!
//! - A name without the `file:` scheme is a path, used as given (relative to the working
//!   directory, or absolute).
//! - `file:///abs/path` and `file://localhost/abs/path` name `/abs/path`; on Windows
//!   `file:///C:/data/x.csv` names `C:/data/x.csv`. `file:/abs/path` (no authority) names the
//!   same.
//! - Any other host is refused: only local files are read. So is a path that names a network
//!   share (`file:////server/share`), and a `file:` URL whose path is not absolute.
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
    /// The path names a network share (`//server/share`, `\\server\share`), which is another
    /// host however the URL spells it.
    #[error("the file URL '{url}' names a network path: only local files are read")]
    NetworkPath {
        /// The URL as given.
        url: String,
    },
    /// The URL's path is not absolute (`file:data.csv`): a file URL names an absolute path.
    #[error(
        "the file URL '{url}' names no absolute path: write file:///path, or give a relative \
         path without file:"
    )]
    NotAbsolute {
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
/// a path whose escapes do not decode, [`FileUrlError::NetworkPath`] for a path that names a
/// network share, and [`FileUrlError::NotAbsolute`] for a URL whose path is not absolute.
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
    // `file:////server/share` and `file://localhost//server/share` leave a path that a
    // system opens as a share on another host.
    if decoded.starts_with("//") || decoded.starts_with('\\') || decoded.starts_with("/\\") {
        return Err(FileUrlError::NetworkPath {
            url: name.to_string(),
        });
    }
    if !decoded.starts_with('/') && !starts_with_drive(&decoded) {
        return Err(FileUrlError::NotAbsolute {
            url: name.to_string(),
        });
    }
    Ok(PathBuf::from(as_local(decoded)))
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

/// On Windows, a URL path with a drive (`/C:/x`, `/C|/x`, `C|/x`) as the path `C:/x`;
/// elsewhere unchanged.
fn as_local(path: String) -> String {
    if !cfg!(windows) {
        return path;
    }
    let drive = path.strip_prefix('/').unwrap_or(&path);
    if starts_with_drive(drive) {
        let mut local = String::with_capacity(drive.len());
        local.push_str(&drive[..1]);
        local.push(':');
        local.push_str(&drive[2..]);
        local
    } else {
        path
    }
}

/// Whether `path` reads `X:` or `X|` for a drive letter `X`, or the same after one `/`.
fn starts_with_drive(path: &str) -> bool {
    let bytes = path.strip_prefix('/').unwrap_or(path).as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && (bytes[1] == b':' || bytes[1] == b'|')
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
        let piped = if cfg!(windows) {
            "C:/data/people.csv"
        } else {
            "/C|/data/people.csv"
        };
        assert_eq!(
            path("file:///C|/data/people.csv"),
            piped,
            "the old `|` drive form"
        );
    }

    #[test]
    fn a_network_path_is_refused_however_the_url_spells_it() {
        for url in [
            "file:////server/share/people.csv",
            "file://localhost//server/share/people.csv",
            "file:///%5C%5Cserver%5Cshare%5Cpeople.csv",
        ] {
            assert!(
                matches!(local_path(url), Err(FileUrlError::NetworkPath { .. })),
                "{url}: {:?}",
                local_path(url)
            );
        }
    }

    #[test]
    fn a_url_without_an_absolute_path_is_refused() {
        for url in [
            "file:people.csv",
            "file:data/people.csv",
            "file://localhost",
        ] {
            assert!(
                matches!(local_path(url), Err(FileUrlError::NotAbsolute { .. })),
                "{url}: {:?}",
                local_path(url)
            );
        }
        // A drive is absolute on Windows, in either spelling.
        assert!(local_path("file:C:/data/people.csv").is_ok());
        assert!(local_path("file:C|/data/people.csv").is_ok());
    }
}
