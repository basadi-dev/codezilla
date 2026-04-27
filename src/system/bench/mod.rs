/// Codezilla Benchmark Framework
///
/// Provides the infrastructure for running reproducible evaluations of the
/// Codezilla agent against a library of self-contained coding tasks.
///
/// ## Architecture
///
/// The benchmark runner drives `codezilla exec --json --ephemeral` as a
/// subprocess for each task, collects the JSONL event stream, runs validation,
/// and produces structured results.
///
/// Tasks are defined as YAML files under `bench/tasks/<task-id>/task.yaml`.
pub mod runner;
pub mod task;
