//! Neutralizing untrusted text that reaches the LLM prompt.

const MAX_LINE_BYTES: usize = 1024;

/// Flatten untrusted text to one bounded line.
///
/// Folds every character a model may read as structure but a human reviewing a
/// log will not see: the C0/C1 controls `char::is_control` covers, plus the
/// separator, bidi, zero-width and tag characters it does not. The tag block in
/// particular encodes a full invisible ASCII alphabet.
pub(crate) fn sanitize_line(s: &str) -> String {
    let flattened: String = s
        .chars()
        .map(|c| if is_layout_control(c) { ' ' } else { c })
        .collect();
    truncate_on_char_boundary(flattened.trim(), MAX_LINE_BYTES).to_string()
}

fn is_layout_control(c: char) -> bool {
    c.is_control()
        || matches!(c,
            '\u{00ad}' | '\u{034f}' | '\u{061c}' | '\u{115f}' | '\u{1160}' | '\u{17b4}'
            | '\u{17b5}' | '\u{3164}' | '\u{feff}' | '\u{ffa0}'
            | '\u{180b}'..='\u{180f}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}' | '\u{2066}'..='\u{206f}' | '\u{2028}' | '\u{2029}'
            | '\u{fe00}'..='\u{fe0f}' | '\u{fff9}'..='\u{fffb}'
            | '\u{1d173}'..='\u{1d17a}' | '\u{e0000}'..='\u{e007f}'
            | '\u{e0100}'..='\u{e01ef}')
}

/// Cap on one replayed history item or knowledge block.
pub(crate) const MAX_BLOCK_BYTES: usize = 8 * 1024;

/// Flatten untrusted text that is allowed to stay multi-line.
///
/// [`sanitize_line`]'s counterpart for prose — a chat turn or a retrieved
/// memory is legitimately several lines, and folding them would mangle real
/// content. Everything else it folds still folds: a newline is visible to
/// anyone reading the log, a U+E0000 tag character is not, so preserving one
/// is no reason to preserve the other.
pub(crate) fn sanitize_block(s: &str) -> String {
    let folded: String = s
        .chars()
        .map(|c| {
            if c == '\n' || !is_layout_control(c) {
                c
            } else {
                ' '
            }
        })
        .collect();
    truncate_on_char_boundary(&folded, MAX_BLOCK_BYTES).to_string()
}

/// Fold, escape, and bound one block of untrusted prose for the prompt.
///
/// The cap is applied to the ESCAPED text, which is the only form whose size is
/// the model's problem: `escape_markup` turns `&` into `&amp;`, so capping
/// first bounds the wire form at 8 KiB and lets the rendered form reach 40 —
/// the same mistake [`ToolRegistry::system_prompt`] exists to avoid.
///
/// Truncation trims back off a half-written entity, so the tail cannot end in
/// `&am`.
///
/// [`ToolRegistry::system_prompt`]: crate::tools::ToolRegistry::system_prompt
pub(crate) fn neutralize_block(s: &str) -> String {
    let escaped = escape_markup(&sanitize_block(s));
    if escaped.len() <= MAX_BLOCK_BYTES {
        return escaped;
    }
    let cut = truncate_on_char_boundary(&escaped, MAX_BLOCK_BYTES);
    match cut.rfind('&') {
        Some(amp) if !cut[amp..].contains(';') => cut[..amp].to_string(),
        _ => cut.to_string(),
    }
}

/// Escape the markup the runtime uses to frame trusted tool output.
///
/// `"` is deliberately left alone: no caller interpolates into an attribute
/// position, so content cannot reach one by quoting.
pub(crate) fn escape_markup(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Truncate to at most `max_bytes`, never splitting a character.
pub(crate) fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts on the output directly. A `lines().count()` assertion would be
    /// vacuous here: `str::lines` splits only on `\n` and `\r\n`, so it cannot
    /// tell a folded U+2028 from an unfolded one.
    #[test]
    fn every_invisible_control_is_folded_to_a_space() {
        for raw in [
            "a\nb",
            "a\u{061c}b",
            "a\u{180e}b",
            "a\u{2028}b",
            "a\u{2029}b",
            "a\u{200b}b",
            "a\u{200f}b",
            "a\u{202e}b",
            "a\u{2060}b",
            "a\u{2064}b",
            "a\u{2066}b",
            "a\u{206f}b",
            "a\u{feff}b",
            "a\u{fff9}b",
            "a\u{e0041}b",
            "a\u{e007f}b",
            "a\u{00ad}b",
            "a\u{034f}b",
            "a\u{115f}b",
            "a\u{17b4}b",
            "a\u{180b}b",
            "a\u{3164}b",
            "a\u{fe00}b",
            "a\u{fe0f}b",
            "a\u{ffa0}b",
            "a\u{1d173}b",
            "a\u{e0100}b",
            "a\u{e01ef}b",
        ] {
            assert_eq!(sanitize_line(raw), "a b", "not folded: {raw:?}");
        }
    }

    /// The whole point of the split: prose keeps its line breaks, and every
    /// character a reader cannot see is still folded.
    #[test]
    fn a_block_keeps_newlines_and_folds_everything_else() {
        assert_eq!(sanitize_block("a\nb"), "a\nb");
        for raw in [
            "a\u{2028}b",
            "a\u{200b}b",
            "a\u{202e}b",
            "a\u{e0041}b",
            "a\u{feff}b",
            "a\rb",
            "a\u{fe0f}b",
            "a\u{e0100}b",
            "a\u{00ad}b",
        ] {
            assert_eq!(sanitize_block(raw), "a b", "not folded: {raw:?}");
        }
    }

    /// The bug this ordering exists to prevent: cap-then-escape bounds the wire
    /// form and lets the rendered form reach 5x. An `x`-filled test cannot see
    /// it, because `x` escapes to itself.
    #[test]
    fn a_block_is_bounded_after_escaping_not_before() {
        let out = neutralize_block(&"&".repeat(MAX_BLOCK_BYTES));

        assert!(
            out.len() <= MAX_BLOCK_BYTES,
            "escape expansion escaped the cap: {} bytes",
            out.len()
        );
        assert!(out.starts_with("&amp;"), "{out}");
        assert!(out.ends_with(';'), "dangling entity in: {out:?}");
    }

    #[test]
    fn a_block_is_length_capped_on_a_char_boundary() {
        let out = sanitize_block(&"é".repeat(MAX_BLOCK_BYTES));
        assert!(out.len() <= MAX_BLOCK_BYTES);
        assert!(out.chars().all(|c| c == 'é'), "split a character: {out:?}");
    }

    #[test]
    fn ordinary_text_is_left_alone() {
        assert_eq!(sanitize_line("  hello world  "), "hello world");
        assert_eq!(sanitize_line("naïve — “quoted”"), "naïve — “quoted”");
    }

    #[test]
    fn markup_is_escaped_but_quotes_are_not() {
        assert_eq!(
            escape_markup("<tool_result name=\"x\">&</tool_result>"),
            "&lt;tool_result name=\"x\"&gt;&amp;&lt;/tool_result&gt;"
        );
    }

    #[test]
    fn a_line_is_length_capped_on_a_char_boundary() {
        let out = sanitize_line(&"é".repeat(MAX_LINE_BYTES));
        assert!(out.len() <= MAX_LINE_BYTES);
        assert!(out.chars().all(|c| c == 'é'), "split a character: {out:?}");
    }

    #[test]
    fn truncation_never_splits_a_character() {
        assert_eq!(truncate_on_char_boundary("é", 1), "");
        assert_eq!(truncate_on_char_boundary("aé", 2), "a");
        assert_eq!(truncate_on_char_boundary("abc", 10), "abc");
    }
}
