//! The options of a text index: its BM25 parameters, its tokenizer and its
//! stop words.
//!
//! They are plain data, in every build: a database reads and writes the
//! definition of a text index (the catalog keeps these options with it) also
//! in a build without the `text-index` feature, which then refuses to open
//! it rather than lose it.

use std::fmt;

use grafeo_common::utils::error::{Error, Result};

/// The most bytes the stop words of one text index may hold in all: 1 MiB,
/// so the catalog record of the index (at most 2 MiB) always holds them.
pub const MAX_STOP_WORD_BYTES: usize = 1 << 20;

/// Configuration for BM25 scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct BM25Config {
    /// Term frequency saturation parameter (default 1.2).
    ///
    /// Higher values give more weight to term frequency; 0 ignores it.
    pub k1: f64,
    /// Length normalization parameter (default 0.75).
    ///
    /// 0.0 = no length normalization, 1.0 = full normalization.
    pub b: f64,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl BM25Config {
    /// Checks the parameters: `k1` finite and at least 0, `b` from 0 to 1.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] naming the parameter out of range.
    pub fn check(&self) -> Result<()> {
        if !self.k1.is_finite() || self.k1 < 0.0 {
            return Err(Error::InvalidValue(format!(
                "the BM25 parameter k1 must be a finite number of at least 0, not {}",
                self.k1
            )));
        }
        if !(0.0..=1.0).contains(&self.b) {
            return Err(Error::InvalidValue(format!(
                "the BM25 parameter b must be a number from 0 to 1, not {}",
                self.b
            )));
        }
        Ok(())
    }
}

/// How a text index splits text into terms. Every tokenizer splits on the
/// characters that are not Unicode letters or digits and lowercases what is
/// left; queries are split the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TokenizerKind {
    /// The default: terms of at least 2 bytes, without about 70 common
    /// English words ("the", "and", ...).
    #[default]
    Simple,
    /// Every term, of any length, without stop words: for languages that
    /// separate words with spaces or punctuation (Russian, Greek, Arabic,
    /// ...).
    Standard,
    /// As [`Standard`](Self::Standard), and every run of Chinese, Japanese
    /// or Korean characters becomes its overlapping pairs of characters
    /// ("柏林市" gives "柏林" and "林市"; a single character stays itself), as
    /// the CJK analyzers of Lucene and Elasticsearch do: for text without
    /// spaces between words.
    CjkBigram,
}

impl TokenizerKind {
    /// Every tokenizer, in the order of their names in errors.
    pub const ALL: [Self; 3] = [Self::Simple, Self::Standard, Self::CjkBigram];

    /// The tokenizer's name: `simple`, `standard` or `cjk_bigram`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Standard => "standard",
            Self::CjkBigram => "cjk_bigram",
        }
    }

    /// The tokenizer named `name` (in any case), or `None`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.name().eq_ignore_ascii_case(name))
    }

    /// The tokenizer named `name` (in any case).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] naming the tokenizers there are.
    pub fn parse(name: &str) -> Result<Self> {
        Self::from_name(name).ok_or_else(|| {
            let names: Vec<&str> = Self::ALL.into_iter().map(Self::name).collect();
            Error::InvalidValue(format!(
                "Unknown tokenizer '{name}'. Use: {}",
                names.join(", ")
            ))
        })
    }
}

impl fmt::Display for TokenizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The options of a text index: its BM25 parameters, its tokenizer and its
/// stop words. The default is BM25 with k1 1.2 and b 0.75, the
/// [`simple`](TokenizerKind::Simple) tokenizer and its English stop words.
///
/// ```
/// use grafeo_core::index::text::{TextIndexOptions, TokenizerKind};
///
/// let options = TextIndexOptions::new()
///     .with_k1(1.5)
///     .with_b(0.3)
///     .with_tokenizer(TokenizerKind::Standard)
///     .with_stop_words(["и", "В"]);
/// assert_eq!(options.stop_words(), Some(&["в".to_string(), "и".to_string()][..]));
/// assert!(options.check().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct TextIndexOptions {
    bm25: BM25Config,
    tokenizer: TokenizerKind,
    stop_words: Option<Vec<String>>,
}

impl TextIndexOptions {
    /// The default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The options a binding or a statement names part by part: each part
    /// the default when `None`, the tokenizer by its name (see
    /// [`TokenizerKind::parse`]); checked (see [`check`](Self::check)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] for an unknown tokenizer and for
    /// options out of range.
    pub fn from_parts(
        k1: Option<f64>,
        b: Option<f64>,
        tokenizer: Option<&str>,
        stop_words: Option<&[String]>,
    ) -> Result<Self> {
        let mut options = Self::new();
        if let Some(k1) = k1 {
            options = options.with_k1(k1);
        }
        if let Some(b) = b {
            options = options.with_b(b);
        }
        if let Some(name) = tokenizer {
            options = options.with_tokenizer(TokenizerKind::parse(name)?);
        }
        if let Some(words) = stop_words {
            options = options.with_stop_words(words);
        }
        options.check()?;
        Ok(options)
    }

    /// Sets the BM25 parameter k1 (term frequency saturation).
    #[must_use]
    pub fn with_k1(mut self, k1: f64) -> Self {
        self.bm25.k1 = k1;
        self
    }

    /// Sets the BM25 parameter b (length normalization).
    #[must_use]
    pub fn with_b(mut self, b: f64) -> Self {
        self.bm25.b = b;
        self
    }

    /// Sets both BM25 parameters.
    #[must_use]
    pub fn with_bm25(mut self, bm25: BM25Config) -> Self {
        self.bm25 = bm25;
        self
    }

    /// Sets the tokenizer.
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: TokenizerKind) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Sets the stop words: terms the index leaves out of documents and
    /// queries, in place of the tokenizer's own (the English words of
    /// [`simple`](TokenizerKind::Simple); the others have none). They are
    /// kept lowercased, sorted and once each, as the tokenizer compares
    /// lowercased terms; empty words are left out. An empty list leaves no
    /// stop words at all.
    #[must_use]
    pub fn with_stop_words<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut words: Vec<String> = words
            .into_iter()
            .map(|word| word.as_ref().to_lowercase())
            .filter(|word| !word.is_empty())
            .collect();
        words.sort_unstable();
        words.dedup();
        self.stop_words = Some(words);
        self
    }

    /// The BM25 parameter k1.
    #[must_use]
    pub fn k1(&self) -> f64 {
        self.bm25.k1
    }

    /// The BM25 parameter b.
    #[must_use]
    pub fn b(&self) -> f64 {
        self.bm25.b
    }

    /// The BM25 parameters.
    #[must_use]
    pub fn bm25(&self) -> &BM25Config {
        &self.bm25
    }

    /// The tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> TokenizerKind {
        self.tokenizer
    }

    /// The stop words set with [`with_stop_words`](Self::with_stop_words),
    /// or `None` for the tokenizer's own.
    #[must_use]
    pub fn stop_words(&self) -> Option<&[String]> {
        self.stop_words.as_deref()
    }

    /// Checks the options: the BM25 parameters (see [`BM25Config::check`])
    /// and the size of the stop words, at most [`MAX_STOP_WORD_BYTES`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidValue`] naming what is out of range.
    pub fn check(&self) -> Result<()> {
        self.bm25.check()?;
        let bytes: usize = self
            .stop_words()
            .unwrap_or_default()
            .iter()
            .map(String::len)
            .sum();
        if bytes > MAX_STOP_WORD_BYTES {
            return Err(Error::InvalidValue(format!(
                "the stop words of a text index hold at most {MAX_STOP_WORD_BYTES} bytes, \
                 these hold {bytes}"
            )));
        }
        Ok(())
    }

    /// Sets the BM25 parameters a restored index scores with.
    #[cfg(feature = "text-index")]
    pub(crate) fn set_bm25(&mut self, bm25: BM25Config) {
        self.bm25 = bm25;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_options_are_bm25_1_2_and_0_75_with_the_simple_tokenizer() {
        let options = TextIndexOptions::new();
        assert_eq!((options.k1(), options.b()), (1.2, 0.75));
        assert_eq!(options.tokenizer(), TokenizerKind::Simple);
        assert_eq!(options.stop_words(), None, "the tokenizer's own");
        assert!(options.check().is_ok());
    }

    #[test]
    fn k1_and_b_out_of_range_are_invalid_values() {
        for (k1, b, named) in [
            (-0.3, 0.75, "k1"),
            (f64::NAN, 0.75, "k1"),
            (f64::INFINITY, 0.75, "k1"),
            (1.2, -0.19, "b"),
            (1.2, 1.88, "b"),
            (1.2, f64::NAN, "b"),
        ] {
            let error = TextIndexOptions::new()
                .with_k1(k1)
                .with_b(b)
                .check()
                .unwrap_err();
            assert!(
                matches!(&error, Error::InvalidValue(message) if message.contains(named)),
                "k1 {k1}, b {b}: {error}"
            );
            assert_eq!(error.error_code().as_str(), "GRAFEO-V001", "{error}");
        }
        for (k1, b) in [(0.0, 0.0), (0.0, 1.0), (88.0, 0.3)] {
            assert!(
                TextIndexOptions::new()
                    .with_k1(k1)
                    .with_b(b)
                    .check()
                    .is_ok(),
                "k1 {k1}, b {b} are in range"
            );
        }
    }

    #[test]
    fn stop_words_are_kept_lowercased_sorted_and_once() {
        let options = TextIndexOptions::new().with_stop_words(["Und", "", "der", "UND", "Σ"]);
        assert_eq!(
            options.stop_words(),
            Some(&["der".to_string(), "und".to_string(), "σ".to_string()][..])
        );
        let none = TextIndexOptions::new().with_stop_words(Vec::<String>::new());
        assert_eq!(
            none.stop_words(),
            Some(&[][..]),
            "an empty list, not the default"
        );
    }

    #[test]
    fn stop_words_past_the_limit_are_an_invalid_value() {
        let word = "a".repeat(MAX_STOP_WORD_BYTES / 2);
        let options = TextIndexOptions::new().with_stop_words([word.clone(), format!("{word}b")]);
        let error = options.check().unwrap_err();
        assert!(
            matches!(&error, Error::InvalidValue(message) if message.contains("stop words")),
            "{error}"
        );
        assert!(
            TextIndexOptions::new()
                .with_stop_words([word])
                .check()
                .is_ok(),
            "exactly half the limit"
        );
    }

    #[test]
    fn options_from_parts_are_the_defaults_with_each_part_given() {
        assert_eq!(
            TextIndexOptions::from_parts(None, None, None, None).unwrap(),
            TextIndexOptions::new()
        );
        let words = ["Und".to_string()];
        assert_eq!(
            TextIndexOptions::from_parts(Some(0.3), Some(0.19), Some("Standard"), Some(&words))
                .unwrap(),
            TextIndexOptions::new()
                .with_k1(0.3)
                .with_b(0.19)
                .with_tokenizer(TokenizerKind::Standard)
                .with_stop_words(["und"])
        );
        for (k1, b, tokenizer) in [
            (Some(-3.0), None, None),
            (None, Some(19.0), None),
            (None, None, Some("jieba")),
        ] {
            let error = TextIndexOptions::from_parts(k1, b, tokenizer, None).unwrap_err();
            assert_eq!(error.error_code().as_str(), "GRAFEO-V001", "{error}");
        }
    }

    #[test]
    fn tokenizers_are_named_in_any_case() {
        for kind in TokenizerKind::ALL {
            assert_eq!(TokenizerKind::from_name(kind.name()), Some(kind));
            assert_eq!(
                TokenizerKind::parse(&kind.name().to_uppercase()).unwrap(),
                kind
            );
        }
        let error = TokenizerKind::parse("jieba").unwrap_err();
        assert_eq!(
            error.to_string(),
            "GRAFEO-V001: Invalid value: Unknown tokenizer 'jieba'. Use: simple, standard, \
             cjk_bigram"
        );
    }
}
