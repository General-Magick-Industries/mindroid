use std::fmt;
use std::sync::{Arc, RwLock};

use super::Tool;

/// A swappable [`ToolRegistry`] handle: readers get a consistent snapshot via
/// [`load`](DynamicRegistry::load); a manifest swaps in a fresh registry via
/// [`store`](DynamicRegistry::store). Writes are rare (manifest/reconnect),
/// reads are per-turn, so a plain `RwLock<Arc<..>>` suffices.
#[derive(Clone)]
pub struct DynamicRegistry {
    inner: Arc<RwLock<Arc<ToolRegistry>>>,
}

impl DynamicRegistry {
    pub fn new(registry: ToolRegistry) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Arc::new(registry))),
        }
    }

    /// Current registry snapshot.
    pub fn load(&self) -> Arc<ToolRegistry> {
        self.inner.read().unwrap().clone()
    }

    /// Replace the registry.
    pub fn store(&self, registry: ToolRegistry) {
        *self.inner.write().unwrap() = Arc::new(registry);
    }
}

/// Registry of tools available to the agent.
///
/// Pass to [`ToolExecutorStage`](crate::pipeline::stages::ToolExecutorStage) after building:
///
/// ```ignore
/// let registry = ToolRegistry::new()
///     .register(ShellTool::default())
///     .register(OpenTool);
/// let stage = ToolExecutorStage::new(client, Arc::new(registry));
/// ```
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Register a tool. Consumes and returns `self` for chaining.
    pub fn register(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Rebuild with the local (non-remote) tools kept and the remote set
    /// replaced by `remote`. Used to apply a client-advertised tool manifest at
    /// runtime without dropping the agent's built-in tools.
    pub fn with_remote_tools(&self, remote: Vec<Arc<dyn Tool>>) -> Self {
        let mut tools: Vec<Arc<dyn Tool>> = self
            .tools
            .iter()
            .filter(|t| !t.is_remote())
            .cloned()
            .collect();
        tools.extend(remote);
        Self { tools }
    }

    /// Additively extend with `extra` tools, keeping ALL existing tools. A tool
    /// whose name already exists is skipped (existing wins). Used for per-turn
    /// tools layered onto the persistent snapshot without dropping anything.
    pub fn plus_tools(&self, extra: Vec<Arc<dyn Tool>>) -> Self {
        let mut tools = self.tools.clone();
        for t in extra {
            if !tools.iter().any(|e| e.name() == t.name()) {
                tools.push(t);
            }
        }
        Self { tools }
    }

    /// Look up a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Build the system prompt fragment that describes all registered tools to the LLM.
    pub fn system_prompt(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut out = String::from(
            "You can control this computer using tools.\n\
             To use a tool, output EXACTLY this on its own line:\n\n\
             <tool_call>{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}</tool_call>\n\n\
             You will receive a <tool_result> back, then can call another tool or give your final answer.\n\
             Keep shell commands short and direct — avoid complex arithmetic or long pipelines in one command; use separate tool calls instead.\n\n\
             Available tools:\n",
        );

        for tool in &self.tools {
            let schema = tool.parameters_schema();
            let params = schema
                .get("properties")
                .and_then(|p| p.as_object())
                .map(|props| {
                    props
                        .iter()
                        .map(|(k, v)| {
                            let desc = v.get("description").and_then(|d| d.as_str()).unwrap_or("");
                            format!("  - {k}: {desc}")
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            out.push_str(&format!(
                "\n**{}** — {}\n{}\n",
                tool.name(),
                tool.description(),
                params
            ));
        }

        out
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &format!("{} tools", self.tools.len()))
            .finish()
    }
}

/// Build a registry from config, enabling only the tools that are configured on.
impl ToolRegistry {
    pub fn from_config(config: &crate::config::ToolsConfig) -> Self {
        let mut registry = Self::new();
        if config.shell.enabled {
            registry = registry.register(super::ShellTool::with_config(
                config.shell.timeout_secs,
                config.shell.instructions.clone(),
                config.shell.clone(),
            ));
        }
        if config.open.enabled {
            registry = registry.register(super::OpenTool::with_config(config.open.clone()));
        }
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::tools::ToolContext;
    use async_trait::async_trait;
    use serde_json::{Value, json};

    struct NamedTool {
        name: String,
        remote: bool,
    }

    impl NamedTool {
        fn local(name: &str) -> Arc<dyn Tool> {
            Arc::new(Self {
                name: name.into(),
                remote: false,
            })
        }
        fn remote(name: &str) -> Arc<dyn Tool> {
            Arc::new(Self {
                name: name.into(),
                remote: true,
            })
        }
    }

    #[async_trait]
    impl Tool for NamedTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn parameters_schema(&self) -> Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(&self, _args: Value, _ctx: &ToolContext) -> Result<String> {
            Ok("ok".into())
        }
        fn is_remote(&self) -> bool {
            self.remote
        }
    }

    fn names(r: &ToolRegistry) -> Vec<&str> {
        r.tools().iter().map(|t| t.name()).collect()
    }

    #[test]
    fn load_returns_the_stored_registry_and_store_replaces_it() {
        let dynamic = DynamicRegistry::new(ToolRegistry::new().register(NamedTool {
            name: "first".into(),
            remote: false,
        }));
        assert_eq!(names(&dynamic.load()), ["first"]);

        dynamic.store(ToolRegistry::new().register(NamedTool {
            name: "second".into(),
            remote: false,
        }));
        assert_eq!(names(&dynamic.load()), ["second"]);
    }

    /// The load-then-swap guarantee: a snapshot taken before a `store` keeps its
    /// own contents, so an executor iterating it mid-turn cannot see it tear.
    #[test]
    fn a_snapshot_is_unaffected_by_a_concurrent_swap() {
        let dynamic = DynamicRegistry::new(ToolRegistry::new().register(NamedTool {
            name: "before".into(),
            remote: false,
        }));

        let snapshot = dynamic.load();
        dynamic.store(ToolRegistry::new().register(NamedTool {
            name: "after".into(),
            remote: false,
        }));

        assert_eq!(
            names(&snapshot),
            ["before"],
            "held snapshot must not change"
        );
        assert_eq!(
            names(&dynamic.load()),
            ["after"],
            "new readers see the swap"
        );
    }

    /// Clones share one slot — a swap through any handle is visible to all.
    #[test]
    fn clones_share_the_same_underlying_slot() {
        let a = DynamicRegistry::new(ToolRegistry::new());
        let b = a.clone();

        b.store(ToolRegistry::new().register(NamedTool {
            name: "via-clone".into(),
            remote: false,
        }));

        assert_eq!(names(&a.load()), ["via-clone"]);
    }

    #[test]
    fn store_is_visible_across_threads() {
        let dynamic = DynamicRegistry::new(ToolRegistry::new());
        let writer = dynamic.clone();

        let h = std::thread::spawn(move || {
            writer.store(ToolRegistry::new().register(NamedTool {
                name: "from-thread".into(),
                remote: false,
            }));
        });
        h.join().unwrap();

        assert_eq!(names(&dynamic.load()), ["from-thread"]);
    }

    #[test]
    fn with_remote_tools_replaces_remotes_but_keeps_local_ones() {
        let base = ToolRegistry {
            tools: vec![NamedTool::local("shell"), NamedTool::remote("old_remote")],
        };

        let updated = base.with_remote_tools(vec![NamedTool::remote("new_remote")]);

        assert_eq!(names(&updated), ["shell", "new_remote"]);
    }

    #[test]
    fn plus_tools_is_additive_and_existing_wins_on_name_collision() {
        let base = ToolRegistry {
            tools: vec![NamedTool::local("shell"), NamedTool::local("open")],
        };

        let updated = base.plus_tools(vec![
            NamedTool::remote("shell"), // collides — must be skipped
            NamedTool::remote("per_turn"),
        ]);

        assert_eq!(names(&updated), ["shell", "open", "per_turn"]);
        assert!(
            !updated.get("shell").unwrap().is_remote(),
            "the pre-existing local tool must survive the collision"
        );
    }

    #[test]
    fn get_and_is_empty_reflect_contents() {
        let empty = ToolRegistry::new();
        assert!(empty.is_empty());
        assert!(empty.get("nope").is_none());

        let one = empty.register(NamedTool {
            name: "shell".into(),
            remote: false,
        });
        assert!(!one.is_empty());
        assert_eq!(one.get("shell").unwrap().name(), "shell");
        assert!(one.get("missing").is_none());
    }
}
