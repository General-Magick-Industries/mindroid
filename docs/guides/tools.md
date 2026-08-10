# Tools System

Tools are capabilities the agent can invoke on the local computer. The LLM decides when and how to use them. While skills provide knowledge (prompt-level), tools provide actions (execution-level).

## Why This Design

LLMs are powerful reasoning engines but cannot act on the world. Tools bridge this gap safely:

- The LLM reasons about *what* to do and emits a structured tool call
- The runtime parses the call, validates it, and executes it in a sandbox
- Results are fed back to the LLM for further reasoning

Mindroid uses XML-based tool calling rather than JSON function calling. XML is more robust with streaming -- partial XML is easier to detect and buffer than partial JSON, and LLMs produce fewer malformed XML tool calls in practice.

---

## Tool Trait

Every tool implements four methods:

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;  // JSON Schema
    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<String>;
}
```

### Custom Tool Example

```rust
use mindroid::tools::{Tool, ToolContext};
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait::async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str { "get_weather" }

    fn description(&self) -> &str {
        "Get the current weather for a city."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        })
    }

    async fn execute(&self, args: Value, _ctx: &ToolContext) -> mindroid::Result<String> {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(format!("Weather in {city}: 22C, sunny"))
    }
}
```

---

## Tool Registry

The registry collects tools and generates the system prompt fragment that teaches the LLM how to call them:

```rust
let registry = ToolRegistry::new()
    .register(ShellTool::default())
    .register(OpenTool)
    .register(WeatherTool);

// Generates XML descriptions of all tools for the system prompt
let tool_prompt = registry.system_prompt();
```

The generated prompt includes each tool's name, description, and JSON Schema parameters. The LLM uses this to know what tools are available and how to call them.

---

## Tool Execution Loop

`ToolExecutorStage` implements an iterative loop that lets the LLM use multiple tools in sequence:

```
┌─────────────────────────────────────────────────┐
│                                                  │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐  │
│  │ Call LLM  │───►│ Parse XML │───►│ Tool     │  │
│  │           │    │ tool calls│    │ calls?   │  │
│  └──────────┘    └───────────┘    └────┬─────┘  │
│       ▲                                │         │
│       │              No ◄──────────────┤         │
│       │                           Yes  │         │
│       │                                ▼         │
│       │                          ┌──────────┐    │
│       │                          │ Execute   │    │
│       │                          │ each tool │    │
│       │                          └─────┬────┘    │
│       │                                │         │
│       │                          ┌─────▼────┐    │
│       └──────────────────────────│ Feed      │    │
│                                  │ results   │    │
│                                  │ to LLM    │    │
│                                  └──────────┘    │
│                                                  │
│  No tool calls ──► Final response                │
└─────────────────────────────────────────────────┘
```

The LLM emits tool calls as XML blocks in its response:

```xml
<tool_call>{"name": "shell", "args": {"command": "ls -la"}}</tool_call>
```

The parser extracts these, executes each tool, and feeds results back as user messages. This repeats until the LLM responds without tool calls (max 20 iterations by default).

### Configuration

```rust
ToolExecutorStage::new(client, registry)
    .with_max_iterations(10)       // default: 20
    .with_parser(MyCustomParser)   // default: XmlToolCallParser
```

### Streaming Behavior

During streaming, the tool executor buffers chunks to detect tool calls before yielding them. This prevents XML tool-call syntax from being spoken aloud or displayed to the user. Only after confirming no tool calls are present does it yield the buffered text.

### Robustness

The XML parser handles common LLM failure modes:
- Malformed JSON inside tool calls (`extract_balanced_json` finds the first valid `{...}`)
- Unclosed strings and braces (`repair_json` closes them)
- Missing closing `</tool_call>` tags

---

## Built-in Tools

### ShellTool

Runs shell commands with multiple safety layers:

**Command allowlist** -- Only permitted commands can run. Defaults to a curated safe set (ls, cat, grep, git, cargo, etc.). Configurable per deployment:

```toml
[tools.shell]
enabled = true
timeout_secs = 30
allowed_commands = ["ls", "cat", "grep", "git"]
```

**Privilege blocking** -- `sudo`, `doas`, `pkexec`, `su` are always rejected regardless of allowlist.

**Environment scrubbing** -- The child process inherits only a safe subset of environment variables (PATH, HOME, USER, LANG, TERM). Secrets like API keys, tokens, and credentials are stripped to prevent accidental leaks.

**Timeout enforcement** -- Commands are killed after `timeout_secs` (default: 30).

**Output truncation** -- Output is capped at 1 MB to prevent context overflow.

**Custom instructions** -- Free-text hints about the system setup can be injected into the tool description so the LLM picks the right commands:

```toml
[tools.shell]
instructions = """
Desktop: i3wm on X11. Screen lock: i3lock.
Media: Spotify (playerctl). Brightness: brightnessctl.
"""
```

### OpenTool

Opens URLs or launches applications using the platform's default handler:
- Linux: `xdg-open`
- macOS: `open`
- Windows: `cmd /C start`

Supports web URLs, app URI schemes (`spotify:collection`), and app names. Optional scheme allowlist for security (default: `http`, `https`).

### SetReminderTool and ReminderRoutine

**SetReminderTool** -- The LLM sets reminders with a message and delay. Stored in a shared `ReminderStore`.

**ReminderRoutine** -- A background `Routine` that checks for due reminders every second and fires them through the transport. Integrates with the runtime's routine system.

```rust
let reminder_store = new_reminder_store();

let tools = ToolRegistry::new()
    .register(SetReminderTool::new(reminder_store.clone()));

let runtime = Runtime::builder()
    .add_routine(ReminderRoutine::new(reminder_store))
    // ...
    .build()?;
```
