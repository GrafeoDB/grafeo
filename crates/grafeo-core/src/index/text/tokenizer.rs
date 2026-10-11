//! Text tokenization for full-text search.

use std::collections::HashSet;

use super::options::{TextIndexOptions, TokenizerKind};

/// A tokenizer splits text into searchable terms.
pub trait Tokenizer: Send + Sync {
    /// Tokenizes text into a list of normalized terms.
    fn tokenize(&self, text: &str) -> Vec<String>;
}

/// The words of `text`: its runs of Unicode letters and digits, lowercased.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

/// Whether `c` is a Chinese, Japanese or Korean character that
/// [`CjkBigramTokenizer`] pairs: a CJK ideograph (with the extensions and
/// the compatibility ideographs, and the marks 々 and 〇), hiragana, katakana
/// (full and half width) or hangul (syllables and jamo).
fn is_cjk(c: char) -> bool {
    matches!(
        u32::from(c),
        0x1100..=0x11FF
            | 0x3005
            | 0x3007
            | 0x3040..=0x30FF
            | 0x3130..=0x318F
            | 0x31F0..=0x31FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA960..=0xA97F
            | 0xAC00..=0xD7FF
            | 0xF900..=0xFAFF
            | 0xFF66..=0xFF9F
            | 0x2_0000..=0x3_134F
    )
}

/// Appends the terms of the CJK run `run` to `terms`: its overlapping pairs
/// of characters, or the character itself when it is alone.
fn push_bigrams(run: &[char], terms: &mut Vec<String>) {
    if let [single] = run {
        terms.push(single.to_string());
    }
    for pair in run.windows(2) {
        terms.push(pair.iter().collect());
    }
}

/// The terms of the [`standard`](TokenizerKind::Standard) tokenizer: every
/// word of `text`.
fn standard_terms(text: &str) -> Vec<String> {
    words(text).collect()
}

/// The terms of the [`cjk_bigram`](TokenizerKind::CjkBigram) tokenizer: every
/// word of `text`, each run of CJK characters in it as its pairs of
/// characters, the rest of the word as it is.
fn cjk_bigram_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for word in words(text) {
        let mut run: Vec<char> = Vec::new();
        let mut other = String::new();
        for c in word.chars() {
            if is_cjk(c) {
                if !other.is_empty() {
                    terms.push(std::mem::take(&mut other));
                }
                run.push(c);
            } else {
                if !run.is_empty() {
                    push_bigrams(&run, &mut terms);
                    run.clear();
                }
                other.push(c);
            }
        }
        if !other.is_empty() {
            terms.push(other);
        }
        if !run.is_empty() {
            push_bigrams(&run, &mut terms);
        }
    }
    terms
}

/// A tokenizer that keeps every word, of any length, lowercased, without
/// stop words: the [`standard`](TokenizerKind::Standard) tokenizer.
///
/// ```
/// use grafeo_core::index::text::{StandardTokenizer, Tokenizer};
///
/// let tokens = StandardTokenizer::new().tokenize("Аликс и Гас едут в Берлин");
/// assert_eq!(tokens, ["аликс", "и", "гас", "едут", "в", "берлин"]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardTokenizer;

impl StandardTokenizer {
    /// Creates the tokenizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for StandardTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        standard_terms(text)
    }
}

/// A tokenizer for text without spaces between words, the
/// [`cjk_bigram`](TokenizerKind::CjkBigram) tokenizer: as
/// [`StandardTokenizer`], with every run of Chinese, Japanese or Korean
/// characters turned into its overlapping pairs of characters.
///
/// ```
/// use grafeo_core::index::text::{CjkBigramTokenizer, Tokenizer};
///
/// let tokens = CjkBigramTokenizer::new().tokenize("住在柏林 Berlin");
/// assert_eq!(tokens, ["住在", "在柏", "柏林", "berlin"]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CjkBigramTokenizer;

impl CjkBigramTokenizer {
    /// Creates the tokenizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for CjkBigramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        cjk_bigram_terms(text)
    }
}

/// The tokenizer of a text index's options: the tokenizer they name, with
/// their stop words in place of its own when they set any.
pub(crate) struct OptionsTokenizer {
    kind: TokenizerKind,
    /// `None`: the tokenizer's own stop words.
    stop_words: Option<HashSet<String>>,
}

impl OptionsTokenizer {
    /// The tokenizer of `options`.
    pub(crate) fn new(options: &TextIndexOptions) -> Self {
        Self {
            kind: options.tokenizer(),
            stop_words: options
                .stop_words()
                .map(|words| words.iter().cloned().collect()),
        }
    }
}

impl Tokenizer for OptionsTokenizer {
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut terms = match self.kind {
            TokenizerKind::Simple => words(text)
                .filter(|word| word.len() >= SIMPLE_MIN_TOKEN_BYTES)
                .collect(),
            TokenizerKind::Standard => standard_terms(text),
            TokenizerKind::CjkBigram => cjk_bigram_terms(text),
        };
        match &self.stop_words {
            Some(stop_words) => terms.retain(|term| !stop_words.contains(term)),
            None if self.kind == TokenizerKind::Simple => {
                terms.retain(|term| !is_english_stop_word(term));
            }
            None => {}
        }
        terms
    }
}

/// The fewest bytes a term of [`SimpleTokenizer::new`] holds.
const SIMPLE_MIN_TOKEN_BYTES: usize = 2;

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
            min_token_length: SIMPLE_MIN_TOKEN_BYTES,
        }
    }

    /// Creates a tokenizer with a custom minimum token length.
    #[must_use]
    pub fn with_min_length(min_token_length: usize) -> Self {
        Self { min_token_length }
    }
}

/// Whether `word` (lowercased) is one of the English stop words of
/// [`SimpleTokenizer`].
fn is_english_stop_word(word: &str) -> bool {
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
            .filter(|s| s.len() >= self.min_token_length && !is_english_stop_word(s))
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

    /// The terms the tokenizer of `options` makes of `text`.
    fn terms(options: &TextIndexOptions, text: &str) -> Vec<String> {
        OptionsTokenizer::new(options).tokenize(text)
    }

    fn with(kind: TokenizerKind) -> TextIndexOptions {
        TextIndexOptions::new().with_tokenizer(kind)
    }

    const RUSSIAN: &str = "Аликс и Гас едут в Берлин";
    const GREEK: &str = "Ο ΓΚΑΣ ζει στο Βερολίνο";
    const CHINESE: &str = "阿利克斯住在柏林";
    const JAPANESE: &str = "ガスは東京とベルリンに住む";
    const KOREAN: &str = "구스는 베를린에 산다";

    #[test]
    fn the_default_options_tokenize_as_the_simple_tokenizer() {
        let simple = SimpleTokenizer::new();
        for text in [
            "The Quick Brown Fox",
            "hello, world! how's it going?",
            "version 2.0 released in 2025",
            "café résumé naïve",
            "a",
            RUSSIAN,
            GREEK,
            CHINESE,
            JAPANESE,
            KOREAN,
        ] {
            assert_eq!(
                terms(&TextIndexOptions::new(), text),
                simple.tokenize(text),
                "{text}"
            );
        }
    }

    #[test]
    fn the_standard_tokenizer_keeps_every_word_lowercased() {
        let standard = with(TokenizerKind::Standard);
        assert_eq!(
            terms(&standard, RUSSIAN),
            ["аликс", "и", "гас", "едут", "в", "берлин"]
        );
        assert_eq!(
            terms(&standard, GREEK),
            ["ο", "γκας", "ζει", "στο", "βερολίνο"],
            "a capital sigma at the end of a word lowercases to the final sigma"
        );
        assert_eq!(
            terms(&standard, "The Gus of Amsterdam, 3 times"),
            ["the", "gus", "of", "amsterdam", "3", "times"],
            "no English stop words, single characters kept"
        );
        assert_eq!(
            terms(&standard, CHINESE),
            [CHINESE],
            "text without spaces is one word"
        );
        assert_eq!(terms(&standard, KOREAN), ["구스는", "베를린에", "산다"]);
    }

    #[test]
    fn the_cjk_bigram_tokenizer_pairs_the_characters_of_cjk_runs() {
        let cjk = with(TokenizerKind::CjkBigram);
        assert_eq!(
            terms(&cjk, CHINESE),
            ["阿利", "利克", "克斯", "斯住", "住在", "在柏", "柏林"]
        );
        assert_eq!(
            terms(&cjk, JAPANESE),
            [
                "ガス", "スは", "は東", "東京", "京と", "とベ", "ベル", "ルリ", "リン", "ンに",
                "に住", "住む"
            ]
        );
        assert_eq!(
            terms(&cjk, KOREAN),
            ["구스", "스는", "베를", "를린", "린에", "산다"]
        );
        assert_eq!(
            terms(&cjk, "Alix 在 Berlin, 2024年"),
            ["alix", "在", "berlin", "2024", "年"],
            "a lone CJK character stays itself; other words stay whole"
        );
        assert_eq!(
            terms(&cjk, RUSSIAN),
            terms(&with(TokenizerKind::Standard), RUSSIAN),
            "without CJK characters it tokenizes as the standard tokenizer"
        );
    }

    #[test]
    fn stop_words_replace_the_tokenizers_own() {
        let russian = with(TokenizerKind::Standard).with_stop_words(["И", "в"]);
        assert_eq!(terms(&russian, RUSSIAN), ["аликс", "гас", "едут", "берлин"]);

        let greek = with(TokenizerKind::Standard).with_stop_words(["ο", "στο"]);
        assert_eq!(terms(&greek, GREEK), ["γκας", "ζει", "βερολίνο"]);

        let chinese = with(TokenizerKind::CjkBigram).with_stop_words(["住在"]);
        assert!(
            !terms(&chinese, CHINESE).contains(&"住在".to_string()),
            "a stop word can be a pair of characters"
        );

        let simple = TextIndexOptions::new().with_stop_words(["gus"]);
        assert_eq!(
            terms(&simple, "The Gus of Amsterdam"),
            ["the", "of", "amsterdam"],
            "the English words are no stop words any more, Gus is"
        );
        let none = TextIndexOptions::new().with_stop_words(Vec::<String>::new());
        assert_eq!(
            terms(&none, "The Gus of Amsterdam"),
            ["the", "gus", "of", "amsterdam"],
            "an empty list leaves no stop words"
        );
    }
}
