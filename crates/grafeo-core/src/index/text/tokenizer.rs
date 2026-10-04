//! Text tokenization for full-text search.

/// A tokenizer splits text into searchable terms.
pub trait Tokenizer: Send + Sync {
    /// Tokenizes text into a list of normalized terms.
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// A simple Unicode-aware tokenizer with stop word removal.
///
/// Splits on non-alphanumeric characters, lowercases, and filters
/// common English stop words.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "text-index")]
/// # {
/// use grafeo_core::index::text::SimpleTokenizer;
/// use grafeo_core::index::text::Tokenizer;
///
/// let tokenizer = SimpleTokenizer::new();
/// let tokens = tokenizer.tokenize("The Quick Brown Fox");
/// assert_eq!(tokens, vec!["quick", "brown", "fox"]);
/// # }
/// ```
pub struct SimpleTokenizer {
    min_token_length: usize,
}

impl SimpleTokenizer {
    /// Creates a new tokenizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_token_length: 2,
        }
    }

    /// Creates a tokenizer with a custom minimum token length.
    #[must_use]
    pub fn with_min_length(min_token_length: usize) -> Self {
        Self { min_token_length }
    }

    fn is_stop_word(word: &str) -> bool {
        matches!(
            word,
            "a" | "an"
                | "and"
                | "are"
                | "as"
                | "at"
                | "be"
                | "been"
                | "but"
                | "by"
                | "can"
                | "do"
                | "for"
                | "from"
                | "had"
                | "has"
                | "have"
                | "he"
                | "her"
                | "his"
                | "how"
                | "i"
                | "if"
                | "in"
                | "into"
                | "is"
                | "it"
                | "its"
                | "just"
                | "me"
                | "my"
                | "no"
                | "nor"
                | "not"
                | "of"
                | "on"
                | "or"
                | "our"
                | "out"
                | "own"
                | "she"
                | "so"
                | "some"
                | "such"
                | "than"
                | "that"
                | "the"
                | "their"
                | "them"
                | "then"
                | "there"
                | "these"
                | "they"
                | "this"
                | "to"
                | "too"
                | "up"
                | "us"
                | "very"
                | "was"
                | "we"
                | "were"
                | "what"
                | "when"
                | "where"
                | "which"
                | "while"
                | "who"
                | "whom"
                | "why"
                | "will"
                | "with"
                | "would"
                | "you"
                | "your"
        )
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SimpleTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .filter(|s| s.len() >= self.min_token_length && !Self::is_stop_word(s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let t = SimpleTokenizer::new();
        assert_eq!(t.tokenize("Hello World"), vec!["hello", "world"]);
    }

    #[test]
    fn test_stop_word_removal() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("the quick brown fox");
        assert_eq!(tokens, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn test_punctuation_split() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("hello, world! how's it going?");
        assert_eq!(tokens, vec!["hello", "world", "going"]);
    }

    #[test]
    fn test_empty_string() {
        let t = SimpleTokenizer::new();
        assert!(t.tokenize("").is_empty(), "expected empty");
    }

    #[test]
    fn test_only_stop_words() {
        let t = SimpleTokenizer::new();
        assert!(t.tokenize("the a an is").is_empty(), "expected empty");
    }

    #[test]
    fn test_unicode() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("café résumé naïve");
        assert_eq!(tokens, vec!["café", "résumé", "naïve"]);
    }

    #[test]
    fn test_min_length_filter() {
        let t = SimpleTokenizer::with_min_length(3);
        let tokens = t.tokenize("go run the big dog");
        assert_eq!(tokens, vec!["run", "big", "dog"]);
    }

    #[test]
    fn test_numbers() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("version 2.0 released in 2025");
        assert_eq!(tokens, vec!["version", "released", "2025"]);
    }

    #[test]
    fn test_mixed_case() {
        let t = SimpleTokenizer::new();
        let tokens = t.tokenize("GrafeoDB is FAST");
        assert_eq!(tokens, vec!["grafeodb", "fast"]);
    }
}
