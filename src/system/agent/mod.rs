pub mod approval;
pub mod checkpoint;
pub mod event_bus;
pub mod executor;
pub mod extensions;
pub mod memory;
pub mod middleware;
pub mod model_gateway;
pub mod permission;
pub mod review;
pub mod sandbox;
pub mod speculative;
pub mod state;
pub mod strategy;
pub mod supervisor;
pub mod tool_wrappers;
pub mod tools;

#[cfg(test)]
pub mod fake_model;

#[allow(unused_imports)]
pub use approval::{ApprovalManager, AutoReviewer};
#[allow(unused_imports)]
pub use event_bus::{EventBus, EventFilter, EventSubscription};
#[allow(unused_imports)]
pub use executor::TurnExecutor;
#[allow(unused_imports)]
pub use extensions::ExtensionManager;
#[allow(unused_imports)]
pub use model_gateway::{ModelDescription, ModelGateway, ModelRequest, ModelStreamEvent};
#[allow(unused_imports)]
pub use permission::PermissionManager;
#[allow(unused_imports)]
pub use sandbox::SandboxManager;
#[allow(unused_imports)]
pub use memory::{
    EmbeddingProvider, MemoryEntry, MemoryHit, MemoryKind, MemoryStats, SemanticMemoryStore,
    SqliteVecMemoryStore,
};
#[allow(unused_imports)]
pub use state::{BranchHandle, InMemoryStateManager, StateManager, StateSnapshot};
#[allow(unused_imports)]
pub use tool_wrappers::{CachingToolProvider, LoggingToolProvider, RateLimitToolProvider};
#[allow(unused_imports)]
pub use tools::{
    BashToolProvider, FileToolProvider, ImageToolProvider, ListDirToolProvider,
    RequestUserInputToolProvider, SearchToolProvider, ShellToolProvider, SpawnAgentToolProvider,
    ToolOrchestrator, ToolProvider, WebToolProvider,
};
