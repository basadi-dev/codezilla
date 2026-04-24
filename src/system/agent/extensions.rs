use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::system::domain::SkillDefinition;

fn fs_entries(path: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(path)? {
        entries.push(entry?.path());
    }
    Ok(entries)
}

pub struct ExtensionManager {
    skills: tokio::sync::RwLock<HashMap<String, SkillDefinition>>,
    plugins: tokio::sync::RwLock<HashMap<String, crate::system::domain::PluginDefinition>>,
    mcp_servers: tokio::sync::RwLock<HashMap<String, crate::system::domain::McpServerDefinition>>,
    connectors: tokio::sync::RwLock<HashMap<String, crate::system::domain::ConnectorDefinition>>,
}

impl ExtensionManager {
    pub fn new() -> Self {
        Self {
            skills: tokio::sync::RwLock::new(HashMap::new()),
            plugins: tokio::sync::RwLock::new(HashMap::new()),
            mcp_servers: tokio::sync::RwLock::new(HashMap::new()),
            connectors: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    pub async fn reload_all(&self, cwd: &str) -> Result<()> {
        let mut skills = HashMap::new();
        let skills_dir = Path::new(cwd).join("skills");
        if skills_dir.exists() {
            for entry in fs_entries(&skills_dir)? {
                if entry.extension().and_then(|s| s.to_str()) == Some("md") {
                    let name = entry
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("skill")
                        .to_string();
                    skills.insert(
                        name.clone(),
                        SkillDefinition {
                            skill_id: name.clone(),
                            name: name.clone(),
                            description: format!("Local skill loaded from {}", entry.display()),
                            root_path: entry.to_string_lossy().to_string(),
                            enabled: true,
                        },
                    );
                }
            }
        }
        *self.skills.write().await = skills;
        Ok(())
    }

    pub async fn list_skills(&self, _cwd: &str) -> Vec<SkillDefinition> {
        self.skills.read().await.values().cloned().collect()
    }

    pub async fn list_plugins(&self) -> Vec<crate::system::domain::PluginDefinition> {
        self.plugins.read().await.values().cloned().collect()
    }

    pub async fn list_mcp_servers(&self) -> Vec<crate::system::domain::McpServerDefinition> {
        self.mcp_servers.read().await.values().cloned().collect()
    }

    pub async fn list_connectors(&self) -> Vec<crate::system::domain::ConnectorDefinition> {
        self.connectors.read().await.values().cloned().collect()
    }
}
