//! String utilities for error messages and suggestions.
//!
//! This module provides functions for generating helpful suggestions
//! in error messages, such as "did you mean X?" when a user typos a label.

/// Computes the Levenshtein edit distance between two strings.
///
/// This is the minimum number of single-character edits (insertions,
/// deletions, or substitutions) required to change one string into the other.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows for space efficiency
    let mut prev = vec![0; n + 1];
    let mut curr = vec![0; n + 1];

    // Initialize first row
    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Finds the most similar string from a list of candidates.
///
/// Returns `Some((best_match, distance))` if a close match is found,
/// `None` if no candidates are sufficiently similar.
///
/// A match is considered "close enough" if the edit distance is at most
/// 2 for short strings (<=5 chars) or at most 3 for longer strings.
///
/// # Examples
///
/// ```
/// use grafeo_common::utils::strings::find_similar;
///
/// let candidates = ["Person", "Place", "Organization"];
/// assert_eq!(find_similar("Peron", &candidates), Some("Person"));
/// assert_eq!(find_similar("XYZ", &candidates), None);
/// ```
pub fn find_similar<'a, S: AsRef<str>>(query: &str, candidates: &'a [S]) -> Option<&'a str> {
    if candidates.is_empty() {
        return None;
    }

    // Case-insensitive comparison
    let query_lower = query.to_lowercase();

    let mut best_match: Option<&str> = None;
    let mut best_distance = usize::MAX;

    for candidate in candidates {
        let candidate_str = candidate.as_ref();
        let candidate_lower = candidate_str.to_lowercase();

        // Check for case-insensitive exact match first
        if query_lower == candidate_lower {
            return Some(candidate_str);
        }

        let distance = levenshtein_distance(&query_lower, &candidate_lower);

        if distance < best_distance {
            best_distance = distance;
            best_match = Some(candidate_str);
        }
    }

    // Determine maximum acceptable distance based on query length
    let max_distance = if query.len() <= 3 {
        1 // Very short strings: only 1 edit allowed
    } else if query.len() <= 5 {
        2 // Short strings: 2 edits allowed
    } else {
        3 // Longer strings: 3 edits allowed
    };

    if best_distance <= max_distance {
        best_match
    } else {
        None
    }
}

/// Formats a suggestion hint for error messages.
///
/// # Examples
///
/// ```
/// use grafeo_common::utils::strings::format_suggestion;
///
/// assert_eq!(format_suggestion("Person"), "Did you mean 'Person'?");
/// ```
pub fn format_suggestion(suggestion: &str) -> String {
    format!("Did you mean '{suggestion}'?")
}

/// Formats a list of suggestions as a hint.
///
/// # Examples
///
/// ```
/// use grafeo_common::utils::strings::format_suggestions;
///
/// assert_eq!(
///     format_suggestions(&["Person", "Place"]),
///     "Did you mean one of: 'Person', 'Place'?"
/// );
/// ```
pub fn format_suggestions(suggestions: &[&str]) -> String {
    match suggestions.len() {
        0 => String::new(),
        1 => format_suggestion(suggestions[0]),
        _ => {
            let quoted: Vec<String> = suggestions.iter().map(|s| format!("'{s}'")).collect();
            format!("Did you mean one of: {}?", quoted.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("a", ""), 1);
        assert_eq!(levenshtein_distance("", "a"), 1);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "abd"), 1);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("Person", "Peron"), 1);
        assert_eq!(levenshtein_distance("Person", "person"), 1); // case difference
    }

    #[test]
    fn test_find_similar() {
        let labels = ["Person", "Place", "Organization", "Event"];

        // Typo: missing 's'
        assert_eq!(find_similar("Peron", &labels), Some("Person"));

        // Case insensitive
        assert_eq!(find_similar("person", &labels), Some("Person"));
        assert_eq!(find_similar("PERSON", &labels), Some("Person"));

        // Small typo
        assert_eq!(find_similar("Plase", &labels), Some("Place"));

        // Too different - no suggestion
        assert_eq!(find_similar("XYZ", &labels), None);
        assert_eq!(find_similar("FooBar", &labels), None);

        // Empty candidates
        let empty: Vec<&str> = vec![];
        assert_eq!(find_similar("Person", &empty), None);
    }

    #[test]
    fn test_format_suggestion() {
        assert_eq!(format_suggestion("Person"), "Did you mean 'Person'?");
    }

    #[test]
    fn test_format_suggestions() {
        assert_eq!(format_suggestions(&[]), "");
        assert_eq!(format_suggestions(&["Person"]), "Did you mean 'Person'?");
        assert_eq!(
            format_suggestions(&["Person", "Place"]),
            "Did you mean one of: 'Person', 'Place'?"
        );
    }
}
