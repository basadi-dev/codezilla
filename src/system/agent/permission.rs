use std::path::{Path, PathBuf};

use crate::system::config::EffectiveConfig;
use crate::system::domain::{
    ActionDescriptor, ApprovalCategory, ApprovalPolicyKind, PermissionProfile, SandboxMode,
    SandboxRequest,
};

pub struct PermissionManager {
    trusted_projects: Vec<PathBuf>,
}

impl PermissionManager {
    pub fn new(trusted_projects: &[String]) -> Self {
        Self {
            trusted_projects: trusted_projects.iter().map(PathBuf::from).collect(),
        }
    }

    pub fn resolve_permission_profile(
        &self,
        config: &EffectiveConfig,
        cwd: &str,
    ) -> PermissionProfile {
        let mut profile = config.permission_profile.clone();
        if profile.writable_roots.is_empty()
            && matches!(profile.sandbox_mode, SandboxMode::WorkspaceWrite)
        {
            profile
                .writable_roots
                .push(crate::system::domain::WritableRoot { path: cwd.into() });
        }
        profile
    }

    pub fn requires_approval(
        &self,
        action: &ActionDescriptor,
        policy: &crate::system::domain::ApprovalPolicy,
        cwd: &str,
    ) -> bool {
        match policy.kind {
            ApprovalPolicyKind::Never => false,
            ApprovalPolicyKind::UnlessTrusted => !self.is_trusted(cwd),
            ApprovalPolicyKind::OnFailure => false,
            ApprovalPolicyKind::OnRequest => matches!(
                action.category,
                ApprovalCategory::SandboxEscalation
                    | ApprovalCategory::FileChange
                    | ApprovalCategory::RequestPermissions
                    | ApprovalCategory::ConnectorAction
                    | ApprovalCategory::McpTool
            ),
            ApprovalPolicyKind::Granular => {
                let granular = policy.granular.clone().unwrap_or_default();
                match action.category {
                    ApprovalCategory::SandboxEscalation => granular.sandbox_approval,
                    ApprovalCategory::RulesChange => granular.rules_approval,
                    ApprovalCategory::SkillApproval => granular.skill_approval,
                    ApprovalCategory::RequestPermissions => granular.request_permissions,
                    ApprovalCategory::McpTool => granular.mcp_tool_approval,
                    ApprovalCategory::ConnectorAction => granular.connector_approval,
                    ApprovalCategory::FileChange | ApprovalCategory::Other => true,
                }
            }
        }
    }

    pub fn build_sandbox_request(
        &self,
        action: &ActionDescriptor,
        profile: &PermissionProfile,
    ) -> SandboxRequest {
        let mut writable_roots = profile
            .writable_roots
            .iter()
            .map(|r| r.path.clone())
            .collect::<Vec<_>>();
        writable_roots.extend(action.paths.clone());
        SandboxRequest {
            sandbox_mode: Some(profile.sandbox_mode),
            writable_roots,
            network_enabled: profile.network_enabled,
            allowed_domains: profile.allowed_domains.clone(),
            allowed_unix_sockets: profile.allowed_unix_sockets.clone(),
        }
    }

    fn is_trusted(&self, cwd: &str) -> bool {
        let cwd = Path::new(cwd);
        self.trusted_projects
            .iter()
            .any(|root| cwd.starts_with(root))
    }
}
