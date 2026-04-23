use anyhow::{anyhow, bail, Result};
use clap::{Arg, ArgAction, ArgMatches, Command};
use std::collections::HashMap;
use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use std::process;

mod llm;
mod logger;
mod system;

use system::{
    resolve_paths, AppServer, AuthManager, ConfigManager, ConfigResolutionInput, ExecInvocation,
    ExecServer, ExecSurface, InteractiveInvocation, InteractiveSurface, OutputMode, ProcessContext,
    SerializableProcessContext, ThreadListParams,
};

fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("Fatal error: {info}");
        process::exit(1);
    }));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to create tokio runtime");

    match runtime.block_on(async_main()) {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("Error: {error:#}");
            process::exit(1);
        }
    }
}

async fn async_main() -> Result<i32> {
    let matches = build_cli().get_matches();
    let config_path = matches
        .get_one::<String>("config")
        .map(PathBuf::from)
        .unwrap_or_else(default_config_path);

    if let Some(cd) = matches.get_one::<String>("cd") {
        std::env::set_current_dir(cd)?;
    }

    let cwd = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .to_string_lossy()
        .to_string();
    let paths = resolve_paths(None, config_path.clone());
    let process_context = ProcessContext {
        argv: std::env::args().collect(),
        env: std::env::vars().collect(),
        stdin_is_tty: std::io::stdin().is_terminal(),
        stdout_is_tty: std::io::stdout().is_terminal(),
        stderr_is_tty: std::io::stderr().is_terminal(),
        cwd: cwd.clone(),
        paths: paths.clone(),
    };

    let config_manager = ConfigManager::new(config_path.clone());
    let cli_overrides = build_cli_overrides(&matches)?;
    let effective_config = config_manager.load_effective_config(ConfigResolutionInput {
        process_context: SerializableProcessContext::from(&process_context),
        profile_name: matches.get_one::<String>("profile").cloned(),
        cli_overrides,
        command_defaults: HashMap::new(),
    })?;

    let _log_guard = logger::init(&effective_config)?;
    let auth_manager = AuthManager::new(&paths)?;
    let runtime =
        system::ConversationRuntime::new(effective_config.clone(), auth_manager.get_session())
            .await?;

    match matches.subcommand() {
        Some(("exec", sub)) => {
            let stdin_suffix = read_piped_stdin(process_context.stdin_is_tty)?;
            let surface = ExecSurface::new(
                runtime,
                if sub.get_flag("json") {
                    OutputMode::Jsonl
                } else {
                    OutputMode::Human
                },
                sub.get_one::<String>("output-last-message").cloned(),
            );
            surface
                .start(ExecInvocation {
                    prompt: sub.get_one::<String>("prompt").cloned(),
                    stdin_suffix,
                    output_mode: if sub.get_flag("json") {
                        OutputMode::Jsonl
                    } else {
                        OutputMode::Human
                    },
                    output_last_message_path: sub.get_one::<String>("output-last-message").cloned(),
                    ephemeral: sub.get_flag("ephemeral"),
                    cwd: Some(cwd),
                    thread_id: None,
                })
                .await
        }
        Some(("review", sub)) => {
            let surface = ExecSurface::new(runtime, OutputMode::Human, None);
            surface
                .start(ExecInvocation {
                    prompt: Some(sub.get_one::<String>("prompt").cloned().unwrap_or_else(|| {
                        "Review the current working tree and report findings.".into()
                    })),
                    stdin_suffix: None,
                    output_mode: OutputMode::Human,
                    output_last_message_path: None,
                    ephemeral: true,
                    cwd: Some(cwd),
                    thread_id: None,
                })
                .await
        }
        Some(("login", sub)) => {
            if let Some(token) = sub.get_one::<String>("api-token") {
                auth_manager.login_with_api_token(token.clone())?;
            } else if sub.get_flag("device") {
                auth_manager.login_with_device_code()?;
            } else {
                auth_manager.login_with_browser_session()?;
            }
            println!("Logged in.");
            Ok(0)
        }
        Some(("logout", _)) => {
            auth_manager.logout()?;
            println!("Logged out.");
            Ok(0)
        }
        Some(("app-server", _)) => {
            let server = AppServer::new(runtime);
            server.start_stdio().await?;
            Ok(0)
        }
        Some(("exec-server", _)) => {
            let server = ExecServer::new();
            server.start_stdio().await?;
            Ok(0)
        }
        Some(("resume", sub)) => {
            let thread_id = resolve_resume_target(&runtime, sub).await?;
            let surface = InteractiveSurface::new(runtime);
            surface
                .start(InteractiveInvocation {
                    prompt: None,
                    resume_thread_id: Some(thread_id),
                    fork_thread_id: None,
                    cwd: Some(cwd),
                })
                .await
        }
        Some(("fork", sub)) => {
            let thread_id = resolve_resume_target(&runtime, sub).await?;
            let surface = InteractiveSurface::new(runtime);
            surface
                .start(InteractiveInvocation {
                    prompt: None,
                    resume_thread_id: None,
                    fork_thread_id: Some(thread_id),
                    cwd: Some(cwd),
                })
                .await
        }
        Some(("plugin", _)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.list_plugins().await)?
            );
            Ok(0)
        }
        Some(("mcp", _)) | Some(("mcp-server", _)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.list_mcp_servers().await)?
            );
            Ok(0)
        }
        Some(("sandbox", _)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&effective_config.permission_profile)?
            );
            Ok(0)
        }
        Some(("apply", sub)) => {
            let patch = sub.get_one::<String>("patch").cloned().unwrap_or_default();
            if patch.is_empty() {
                bail!("apply requires --patch");
            }
            std::fs::write("codezilla.patch", patch)?;
            println!("Wrote patch to codezilla.patch");
            Ok(0)
        }
        Some(("features", _)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&effective_config.features)?
            );
            Ok(0)
        }
        Some((command, _)) => bail!("unsupported subcommand: {command}"),
        None => {
            let surface = InteractiveSurface::new(runtime);
            surface
                .start(InteractiveInvocation {
                    prompt: matches.get_one::<String>("prompt").cloned(),
                    resume_thread_id: None,
                    fork_thread_id: None,
                    cwd: Some(cwd),
                })
                .await
        }
    }
}

fn build_cli() -> Command {
    Command::new("codezilla")
        .version(env!("CARGO_PKG_VERSION"))
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("FILE")
                .global(true),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .value_name("MODEL_ID")
                .global(true),
        )
        .arg(
            Arg::new("profile")
                .long("profile")
                .value_name("PROFILE_NAME")
                .global(true),
        )
        .arg(
            Arg::new("sandbox")
                .long("sandbox")
                .value_parser(["read-only", "workspace-write", "danger-full-access"])
                .global(true),
        )
        .arg(
            Arg::new("full-auto")
                .long("full-auto")
                .action(ArgAction::SetTrue)
                .global(true),
        )
        .arg(
            Arg::new("dangerously-bypass-approvals-and-sandbox")
                .long("dangerously-bypass-approvals-and-sandbox")
                .action(ArgAction::SetTrue)
                .conflicts_with("full-auto")
                .global(true),
        )
        .arg(Arg::new("cd").long("cd").value_name("DIR").global(true))
        .arg(
            Arg::new("add-dir")
                .long("add-dir")
                .value_name("DIR")
                .action(ArgAction::Append)
                .global(true),
        )
        .arg(
            Arg::new("image")
                .long("image")
                .value_name("FILE")
                .action(ArgAction::Append)
                .global(true),
        )
        .arg(
            Arg::new("config-override")
                .short('c')
                .value_name("KEY=VALUE")
                .action(ArgAction::Append)
                .global(true),
        )
        .arg(
            Arg::new("enable")
                .long("enable")
                .value_name("FEATURE")
                .action(ArgAction::Append)
                .global(true),
        )
        .arg(
            Arg::new("disable")
                .long("disable")
                .value_name("FEATURE")
                .action(ArgAction::Append)
                .global(true),
        )
        .arg(Arg::new("prompt").value_name("PROMPT"))
        .subcommand(
            Command::new("exec")
                .arg(Arg::new("prompt").value_name("PROMPT"))
                .arg(Arg::new("json").long("json").action(ArgAction::SetTrue))
                .arg(
                    Arg::new("output-last-message")
                        .long("output-last-message")
                        .value_name("FILE"),
                )
                .arg(
                    Arg::new("ephemeral")
                        .long("ephemeral")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("skip-git-repo-check")
                        .long("skip-git-repo-check")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("ignore-user-config")
                        .long("ignore-user-config")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("ignore-rules")
                        .long("ignore-rules")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("output-schema")
                        .long("output-schema")
                        .value_name("FILE"),
                ),
        )
        .subcommand(Command::new("review").arg(Arg::new("prompt").value_name("PROMPT")))
        .subcommand(
            Command::new("login")
                .arg(Arg::new("api-token").long("api-token").value_name("TOKEN"))
                .arg(Arg::new("device").long("device").action(ArgAction::SetTrue)),
        )
        .subcommand(Command::new("logout"))
        .subcommand(Command::new("mcp"))
        .subcommand(Command::new("mcp-server"))
        .subcommand(Command::new("plugin"))
        .subcommand(Command::new("app-server"))
        .subcommand(Command::new("sandbox"))
        .subcommand(Command::new("apply").arg(Arg::new("patch").long("patch").value_name("TEXT")))
        .subcommand(
            Command::new("resume")
                .arg(Arg::new("thread-id").value_name("THREAD_ID"))
                .arg(Arg::new("last").long("last").action(ArgAction::SetTrue)),
        )
        .subcommand(
            Command::new("fork")
                .arg(Arg::new("thread-id").value_name("THREAD_ID"))
                .arg(Arg::new("last").long("last").action(ArgAction::SetTrue)),
        )
        .subcommand(Command::new("exec-server"))
        .subcommand(Command::new("features"))
}

fn build_cli_overrides(matches: &ArgMatches) -> Result<HashMap<String, serde_json::Value>> {
    let mut overrides = HashMap::new();

    if let Some(model) = matches.get_one::<String>("model") {
        overrides.insert(
            "modelSettings.modelId".into(),
            serde_json::Value::String(model.clone()),
        );
    }
    if let Some(sandbox) = matches.get_one::<String>("sandbox") {
        overrides.insert(
            "permissionProfile.sandboxMode".into(),
            serde_json::Value::String(sandbox.clone()),
        );
    }
    if matches.get_flag("full-auto") {
        overrides.insert(
            "approvalPolicy.kind".into(),
            serde_json::Value::String("ON_FAILURE".into()),
        );
    }
    if matches.get_flag("dangerously-bypass-approvals-and-sandbox") {
        overrides.insert(
            "approvalPolicy.kind".into(),
            serde_json::Value::String("NEVER".into()),
        );
        overrides.insert(
            "permissionProfile.sandboxMode".into(),
            serde_json::Value::String("danger-full-access".into()),
        );
    }
    if let Some(add_dirs) = matches.get_many::<String>("add-dir") {
        overrides.insert(
            "addDirs".into(),
            serde_json::Value::Array(
                add_dirs
                    .map(|value| serde_json::Value::String(value.clone()))
                    .collect(),
            ),
        );
    }
    if let Some(features) = matches.get_many::<String>("enable") {
        for feature in features {
            overrides.insert(format!("features.{feature}"), serde_json::Value::Bool(true));
        }
    }
    if let Some(features) = matches.get_many::<String>("disable") {
        for feature in features {
            overrides.insert(
                format!("features.{feature}"),
                serde_json::Value::Bool(false),
            );
        }
    }
    if let Some(values) = matches.get_many::<String>("config-override") {
        for entry in values {
            let (key, raw_value) = entry
                .split_once('=')
                .ok_or_else(|| anyhow!("invalid -c override, expected KEY=VALUE"))?;
            let yaml_value: serde_yaml::Value = serde_yaml::from_str(raw_value)?;
            overrides.insert(key.into(), serde_json::to_value(yaml_value)?);
        }
    }

    Ok(overrides)
}

fn read_piped_stdin(stdin_is_tty: bool) -> Result<Option<String>> {
    if stdin_is_tty {
        return Ok(None);
    }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    if input.is_empty() {
        Ok(None)
    } else {
        Ok(Some(input))
    }
}

async fn resolve_resume_target(
    runtime: &system::ConversationRuntime,
    matches: &ArgMatches,
) -> Result<String> {
    if let Some(thread_id) = matches.get_one::<String>("thread-id") {
        return Ok(thread_id.clone());
    }
    if matches.get_flag("last") {
        return latest_thread_id(runtime).await;
    }
    latest_thread_id(runtime).await
}

async fn latest_thread_id(runtime: &system::ConversationRuntime) -> Result<String> {
    let threads = runtime
        .list_threads(ThreadListParams {
            cwd: None,
            archived: Some(false),
            search_term: None,
            limit: Some(1),
            cursor: None,
        })
        .await?;
    threads
        .threads
        .into_iter()
        .next()
        .map(|thread| thread.thread_id)
        .ok_or_else(|| anyhow!("no threads found"))
}

fn default_config_path() -> PathBuf {
    let local = PathBuf::from("config.yaml");
    if local.exists() {
        return local;
    }
    if let Some(home) = dirs::config_dir() {
        home.join("codezilla").join("config.yaml")
    } else {
        local
    }
}
