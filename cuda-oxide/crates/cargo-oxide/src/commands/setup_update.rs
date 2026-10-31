/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::backend;
use crate::backend_source;
use std::ffi::OsStr;
use std::path::Path;

use super::*;

// =============================================================================
// Setup command
// =============================================================================

/// How `cargo oxide setup` should behave for the current project mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupPlan {
    /// Inside the monorepo: build the local backend source and publish it to
    /// the shared cache.
    BuildAndPublish,
    /// Outside the monorepo: report the backend context resolution already
    /// built or validated for this project's cuda-oxide dependency.
    ReportResolved,
}

pub fn plan_setup(is_workspace: bool) -> SetupPlan {
    if is_workspace {
        SetupPlan::BuildAndPublish
    } else {
        SetupPlan::ReportResolved
    }
}

/// Explicitly build (or rebuild) the codegen backend.
///
/// Normally the backend is built automatically on every `run`/`build`/`pipeline`
/// invocation. `setup` exists for first-time setup, CI, or after pulling new
/// changes when you want to rebuild without running an example.
///
/// Outside the workspace the backend comes from the project's cuda-oxide
/// dependency, which `resolve_context` builds or validates before `setup`
/// runs, so `setup` only reports the result there
pub fn setup(ctx: &Context) {
    match plan_setup(ctx.is_workspace) {
        SetupPlan::BuildAndPublish => build_and_publish(ctx),
        SetupPlan::ReportResolved => report_resolved(ctx),
    }
}

/// Outside the monorepo, `ctx.codegen_crate` is the user's project, not a
/// backend source tree; building it compiles the user's crate and finds no
/// backend artifact (#1688). The source is the project's cuda-oxide
/// dependency, which `resolve_context` already built into the shared cache
/// (in place for a path dependency) or found there with a matching commit.
/// That backend is current by construction, so a second build would only
/// repeat it; `update` is the explicit clear-and-rebuild. A pinned backend is
/// the exception: discovery returns it as soon as the file exists, so setup
/// can't vouch for it
fn report_resolved(ctx: &Context) {
    let env_pin = std::env::var_os("CUDA_OXIDE_BACKEND");
    match backend_pin(
        env_pin.as_deref(),
        ctx.config.backend.as_deref(),
        &ctx.backend_so,
    ) {
        Some(pin) => {
            println!(
                "Using the backend pinned by {pin}: {}",
                ctx.backend_so.display()
            );
            println!("  setup neither builds nor checks a pinned backend. It must have been");
            println!("  built from this project's cuda-oxide commit with the active Rust");
            println!("  toolchain, or rustc fails to load it; unset the pin to have setup");
            println!("  build it from the dependency.");
        }
        None => {
            println!("✓ Backend is ready at {}", ctx.backend_so.display());
            println!("  Built from this project's cuda-oxide dependency, or from cuda-oxide main");
            println!("  when it has none. Run `cargo oxide update` to clear the shared cache and");
            println!("  rebuild it.");
        }
    }
    println!();
    println!("You can now use:");
    println!("  cargo oxide run");
    println!("  cargo oxide build");
}

/// Names the pin that selected `selected`, if one did.
///
/// Compares against the backend discovery actually chose rather than testing
/// whether a pin is set: `find_or_build_backend` warns about a
/// `CUDA_OXIDE_BACKEND` that points at a missing file and falls back to
/// auto-detection, so a set variable doesn't mean it was used
pub fn backend_pin(
    env_backend: Option<&OsStr>,
    configured_backend: Option<&Path>,
    selected: &Path,
) -> Option<&'static str> {
    if env_backend.is_some_and(|path| Path::new(path) == selected) {
        return Some("CUDA_OXIDE_BACKEND");
    }
    if configured_backend == Some(selected) {
        return Some("`.cargo/cuda-oxide.toml`");
    }
    None
}

fn build_and_publish(ctx: &Context) {
    println!("Building cuda-oxide codegen backend...");
    println!();

    let built_so = backend::build_backend_from_source(&ctx.codegen_crate);

    println!();
    println!("✓ Backend is ready. You can now use:");
    println!("  cargo oxide run <example>");
    println!("  cargo oxide build <example>");

    // A project outside this repository resolves the backend through the
    // shared cache, since `find_workspace_root` finds no
    // `crates/rustc-codegen-cuda` above it. Publishing the build there keeps
    // those projects on the backend that was just built instead of on whatever
    // the cache last held. The cache records the commit it came from, so only
    // projects whose cuda-oxide dependency resolves to this checkout's HEAD
    // pick it up; any other project rebuilds from its own dependency.
    match backend::publish_to_cache(&built_so, &ctx.codegen_crate) {
        Some(published) => {
            println!();
            println!("✓ Published to {}", published.path.display());
            match published.source_rev {
                Some(rev) => println!(
                    "  Projects outside this repo whose cuda-oxide dependency resolves to {} will use this build.",
                    backend_source::short_rev(&rev)
                ),
                None => println!(
                    "  Projects outside this repo without a cuda-oxide dependency will use this build."
                ),
            }
        }
        None => {
            eprintln!();
            eprintln!("Warning: could not publish the backend to the shared cache.");
            eprintln!("Projects outside this repo may keep using an older build.");
            eprintln!("Set CUDA_OXIDE_BACKEND to this build to override.");
        }
    }
}

/// How `cargo oxide update` should behave for the current project mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdatePlan {
    /// Inside the monorepo: tell the user to run `setup` (non-destructive).
    AdviseSetup,
    /// Inside the monorepo with `--force`: rebuild via `setup`.
    RunSetup,
    /// Outside the monorepo: clear and rebuild the shared cache.
    RefreshCache,
}

pub fn plan_update(is_workspace: bool, force: bool) -> UpdatePlan {
    match (is_workspace, force) {
        (true, false) => UpdatePlan::AdviseSetup,
        (true, true) => UpdatePlan::RunSetup,
        (false, _) => UpdatePlan::RefreshCache,
    }
}

/// Refresh the codegen backend used by this project.
///
/// Inside the cuda-oxide workspace the authoritative backend is the local
/// source tree, so the default path points at `cargo oxide setup`. Outside
/// the workspace, the shared `~/.cargo/cuda-oxide/` cache is cleared and
/// rebuilt from the commit the project's cuda-oxide dependency resolves to
/// (or from a fresh `main` clone when the project has no such dependency).
/// The backend pin that outranks the shared cache `update` refreshes, if any.
///
/// Both the `CUDA_OXIDE_BACKEND` env var and a `.cargo/cuda-oxide.toml`
/// `backend` entry sit above the cache in backend discovery, so a refreshed
/// cache would never be consulted while either is set. `update` refuses
/// rather than mislead.
fn update_pin_refusal(ctx: &Context) -> Option<String> {
    update_pin_refusal_with_env(ctx, std::env::var_os("CUDA_OXIDE_BACKEND"))
}

/// `update_pin_refusal` with the ambient `CUDA_OXIDE_BACKEND` injected.
///
/// The env var is checked before the project pin, so resolution has to be
/// injectable for unit tests: a developer with `CUDA_OXIDE_BACKEND` exported
/// would otherwise get the env refusal for every input, including the
/// unpinned case that must return `None`. Same rationale as
/// `nvvm_ir_requested_with_env`.
pub(super) fn update_pin_refusal_with_env(
    ctx: &Context,
    backend_env: Option<std::ffi::OsString>,
) -> Option<String> {
    if backend_env.is_some() {
        return Some(
            "CUDA_OXIDE_BACKEND is set, so `cargo oxide update` will not\n\
             modify the shared cache. Unset CUDA_OXIDE_BACKEND and re-run, or\n\
             rebuild the pinned backend path yourself."
                .to_string(),
        );
    }
    ctx.config.backend.as_deref().map(|pinned| {
        format!(
            "`.cargo/cuda-oxide.toml` pins the backend to {}, so\n\
             `cargo oxide update` will not modify the shared cache. Remove the\n\
             `backend` entry and re-run, or rebuild the pinned path yourself.",
            pinned.display()
        )
    })
}

pub fn update(ctx: &Context, force: bool) {
    if let Some(refusal) = update_pin_refusal(ctx) {
        eprintln!("Error: {refusal}");
        std::process::exit(1);
    }

    match plan_update(ctx.is_workspace, force) {
        UpdatePlan::AdviseSetup => {
            println!("Inside the cuda-oxide workspace the codegen backend is built from");
            println!("local source (`crates/rustc-codegen-cuda`).");
            println!();
            println!("Run `cargo oxide setup` to rebuild and publish to the shared cache,");
            println!("or pass `--force` to run setup from this command.");
        }
        UpdatePlan::RunSetup => {
            println!("`--force` requested inside the workspace; running setup...");
            println!();
            setup(ctx);
        }
        UpdatePlan::RefreshCache => {
            // Each route prints its own specific line (cached from the
            // dependency, built in place for a path dependency, or fetched
            // from main), so the framing here stays route-neutral.
            println!("Rebuilding the codegen backend for this project...");
            println!();
            let so = backend::refresh_cached_backend(&ctx.workspace_root);
            println!();
            println!("✓ Backend ready at {}", so.display());
        }
    }
}
