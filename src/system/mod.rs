pub mod agent;
pub mod config;
pub mod domain;
pub mod error;
pub mod mcp;
pub mod persistence;
pub mod runtime;
pub mod server;
pub mod surfaces;
pub mod tui;

pub use config::{
    resolve_paths, AuthManager, ConfigManager, ConfigResolutionInput, ProcessContext,
    SerializableProcessContext,
};
pub use domain::*;
pub use runtime::*;
pub use server::{AppServer, ExecServer};
pub use surfaces::{ExecInvocation, ExecSurface, InteractiveInvocation, InteractiveSurface};
