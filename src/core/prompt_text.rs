//! Neutralizing untrusted text that reaches the LLM prompt.

const MAX_LINE_BYTES: usize = 1024;

/// Flatten untrusted text to one bounded line.
///
/// Beyond `char::is_control` (category Cc) this also folds the separator,
/// bidi and zero-width formatting characters, which a model may render as a
/// line break and which would otherwise let a value forge further entries.
pub(crate) fn sanitize_line(s: &str) -> String {
    let flattened: String = s
        .chars()
        .map(|c| if is_layout_control(c) { ' ' } else { c })
        .collect();
    crate::tools::remote::truncate_on_char_boundary(flattened.trim(), MAX_LINE_BYTES).to_string()
}

fn is_layout_control(c: char) -> bool {
    c.is_control()
        || matches!(c,
            '\u{2028}' | '\u{2029}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}'
            | '\u{2066}'..='\u{2069}' | '\u{feff}')
}

/// Escape the markup the runtime uses to frame trusted tool output.
///
/// `"` is deliberately left alone: every marker this guards is anchored to a
/// tag, so content cannot reach an attribute position by quoting.
pub(crate) fn escape_markup(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newlines_and_unicode_separators_are_flattened() {
        for raw in [
            "a\nb",
            "a\r\nb",
            "a\u{2028}b",
            "a\u{2029}b",
            "a\u{200b}b",
            "a\u{202e}b",
            "a\u{feff}b",
        ] {
            let out = sanitize_line(raw);
            assert_eq!(out.lines().count(), 1, "still one line: {out:?}");
            assert!(!out.contains('\u{2028}'), "separator survived: {out:?}");
        }
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
    }
}
