/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Post-optimization diagnostics for Rust locals that remain stack-backed.
//!
//! Rust MIR locals are tagged before lowering and exported with an encoded SSA
//! name. The ordinary LLVM `opt -O3` pipeline removes locals that mem2reg/SROA
//! can promote. This module inspects the exact LLVM IR subsequently passed to
//! `llc` and reports only tagged allocas that still exist there.
//!
//! The analysis is intentionally conservative. Dynamic GEP indices and obvious
//! address escapes get specific reasons; every other surviving tagged alloca
//! receives a generic "survived scalar replacement" explanation rather than a
//! guessed cause.
//!
//! Most reports are opt-in (verbose builds). One is on by default: a local
//! indexed by a loop counter in a loop LLVM left rolled. That shape has a
//! one-line fix (`#[unroll]`), it is easy to write without noticing (a
//! `for e in 0..4 { r[e] = .. }` fill), and it costs a stack frame and
//! local-memory traffic in every thread.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const LOCAL_MEMORY_VALUE_PREFIX: &str = "%__cuda_oxide_local_x";
const LOCAL_MEMORY_ALLOCA_PREFIX: &str = "%__cuda_oxide_local_alloca_x";

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalProvenance {
    local_index: usize,
    size_bytes: u64,
    name: String,
    type_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromotionBlocker {
    /// A dynamic index derived from a loop induction variable.
    LoopCounterIndex,
    DynamicIndex,
    AddressEscapes,
    SurvivedScalarReplacement,
}

#[derive(Clone, Debug)]
struct FunctionBody<'a> {
    name: String,
    lines: Vec<&'a str>,
}

#[derive(Clone, Debug)]
struct SurvivingAlloca {
    value: String,
    provenance: LocalProvenance,
}

/// Which surviving locals to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reporting {
    /// Only locals indexed by the counter of a rolled loop (default builds).
    LoopCounterIndexOnly,
    /// Every surviving tagged local (verbose builds).
    All,
}

/// Inspect optimized LLVM IR and return user-facing warning diagnostics.
pub(crate) fn diagnose_file(path: &Path, reporting: Reporting) -> std::io::Result<Vec<String>> {
    let llvm = std::fs::read_to_string(path)?;
    Ok(diagnose_text(&llvm, reporting))
}

fn diagnose_text(llvm: &str, reporting: Reporting) -> Vec<String> {
    let mut diagnostics = Vec::new();
    for function in function_bodies(llvm) {
        let function_name = demangle_function_name(&function.name);
        for alloca in surviving_allocas(&function) {
            let blocker = classify_promotion_blocker(&function.lines, &alloca.value);
            if reporting == Reporting::LoopCounterIndexOnly
                && blocker != PromotionBlocker::LoopCounterIndex
            {
                continue;
            }
            diagnostics.push(format_diagnostic(
                &function_name,
                &alloca.provenance,
                blocker,
            ));
        }
    }
    diagnostics
}

/// Render the containing function readably in the warning.
///
/// Kernel entry points already carry their exported source name, but a
/// non-inlined device function survives optimization under its mangled Rust
/// symbol. Demangle those (dropping the disambiguating hash) so the warning
/// points at the source function; anything unmangled passes through untouched.
fn demangle_function_name(name: &str) -> String {
    match rustc_demangle::try_demangle(name) {
        Ok(demangled) => format!("{demangled:#}"),
        Err(_) => name.to_string(),
    }
}

fn function_bodies(llvm: &str) -> Vec<FunctionBody<'_>> {
    let mut functions = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_lines = Vec::new();

    for line in llvm.lines() {
        if current_name.is_none() {
            if let Some(name) = defined_function_name(line) {
                current_name = Some(name);
                current_lines.push(line);
            }
            continue;
        }

        current_lines.push(line);
        if line.trim() == "}" {
            functions.push(FunctionBody {
                name: current_name.take().unwrap(),
                lines: std::mem::take(&mut current_lines),
            });
        }
    }

    functions
}

fn defined_function_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("define ") {
        return None;
    }
    let at = trimmed.find('@')? + 1;
    let rest = &trimmed[at..];
    if let Some(quoted) = rest.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some(quoted[..end].to_string());
    }
    let end = rest.find('(')?;
    Some(rest[..end].trim().to_string())
}

fn surviving_allocas(function: &FunctionBody<'_>) -> Vec<SurvivingAlloca> {
    function
        .lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            if !trimmed.contains(" = alloca ") {
                return None;
            }
            let value = trimmed.split_once(" = ")?.0.trim();
            let provenance = decode_provenance_name(value)?;
            Some(SurvivingAlloca {
                value: value.to_string(),
                provenance,
            })
        })
        .collect()
}

fn decode_provenance_name(value: &str) -> Option<LocalProvenance> {
    let encoded = value
        .strip_prefix(LOCAL_MEMORY_VALUE_PREFIX)
        .or_else(|| value.strip_prefix(LOCAL_MEMORY_ALLOCA_PREFIX))?;
    // The exporter terminates the hex payload with a `_` sentinel. LLVM
    // uniques colliding value names by appending bare digits (a second inlined
    // copy of the same local becomes `<name>1`), and digits are valid hex, so
    // without the sentinel a uniquing suffix would silently extend the payload
    // and garble the report. Key the decode on the sentinel: hex digits up to
    // the first `_`, and reject names that lack it.
    let hex_len = encoded
        .bytes()
        .take_while(|byte| byte.is_ascii_hexdigit())
        .count();
    if encoded.as_bytes().get(hex_len) != Some(&b'_') {
        return None;
    }
    let encoded = &encoded[..hex_len];
    if encoded.is_empty() || encoded.len() % 2 != 0 {
        return None;
    }

    let mut bytes = Vec::with_capacity(encoded.len() / 2);
    for index in (0..encoded.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&encoded[index..index + 2], 16).ok()?);
    }
    let decoded = String::from_utf8(bytes).ok()?;
    let mut fields = decoded.splitn(4, '\t');
    Some(LocalProvenance {
        local_index: fields.next()?.parse().ok()?,
        size_bytes: fields.next()?.parse().ok()?,
        name: fields.next()?.to_string(),
        type_name: fields.next()?.to_string(),
    })
}

fn classify_promotion_blocker(lines: &[&str], root: &str) -> PromotionBlocker {
    let pointer_values = derived_pointer_values(lines, root);
    let dynamic_indices: Vec<&str> = lines
        .iter()
        .flat_map(|line| gep_dynamic_indices(line, &pointer_values))
        .collect();

    if !dynamic_indices.is_empty() {
        let definitions = integer_definitions(lines);
        if dynamic_indices
            .iter()
            .any(|index| derives_from_induction_variable(index, &definitions))
        {
            return PromotionBlocker::LoopCounterIndex;
        }
        return PromotionBlocker::DynamicIndex;
    }

    if lines
        .iter()
        .any(|line| address_escapes(line, &pointer_values))
    {
        return PromotionBlocker::AddressEscapes;
    }

    PromotionBlocker::SurvivedScalarReplacement
}

fn derived_pointer_values(lines: &[&str], root: &str) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    values.insert(root.to_string());

    loop {
        let mut changed = false;
        for line in lines {
            let Some((lhs, rhs)) = assignment(line) else {
                continue;
            };
            if !is_pointer_identity_or_derivation(rhs) {
                continue;
            }
            if values.iter().any(|value| contains_value_token(rhs, value))
                && values.insert(lhs.to_string())
            {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    values
}

fn assignment(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let (lhs, rhs) = trimmed.split_once(" = ")?;
    lhs.starts_with('%').then_some((lhs.trim(), rhs.trim()))
}

fn is_pointer_identity_or_derivation(rhs: &str) -> bool {
    rhs.starts_with("getelementptr ")
        || rhs.starts_with("bitcast ")
        || rhs.starts_with("addrspacecast ")
        || rhs.starts_with("freeze ")
        || rhs.starts_with("phi ")
        || rhs.starts_with("select ")
}

/// The non-constant index operands of a GEP into one of `pointers`.
fn gep_dynamic_indices<'a>(line: &'a str, pointers: &BTreeSet<String>) -> Vec<&'a str> {
    let Some((_, rhs)) = assignment(line) else {
        return Vec::new();
    };
    if !rhs.starts_with("getelementptr ") {
        return Vec::new();
    }

    let Some(root) = pointers
        .iter()
        .find(|value| contains_value_token(rhs, value))
    else {
        return Vec::new();
    };

    let fields: Vec<&str> = rhs.split(',').collect();
    let Some(base_field) = fields
        .iter()
        .position(|field| contains_value_token(field, root))
    else {
        return Vec::new();
    };

    fields[base_field + 1..]
        .iter()
        .flat_map(|field| value_tokens(field))
        .collect()
}

/// One integer SSA definition: its opcode and its value operands.
struct IntegerDefinition<'a> {
    opcode: &'a str,
    operands: Vec<&'a str>,
    /// Whether an operand is an integer literal (an `add %i, 1` step).
    has_literal_operand: bool,
    /// Whether this widens an `i1`: Rust's `Range::next` advances its counter
    /// by `zext i1 %has_next`, not by a literal 1.
    is_bool_zext: bool,
    /// Whether an `icmp` compares this value with a literal: a loop bound
    /// known at compile time, the only kind `#[unroll]` can fully unroll.
    compared_to_literal: bool,
}

/// Opcodes an index can be computed through on its way from a loop counter.
/// Anything else (a load, a call) ends the search: the value is data, not a
/// counter.
const INDEX_ARITHMETIC: &[&str] = &[
    "phi", "add", "sub", "mul", "shl", "lshr", "ashr", "or", "and", "xor", "zext", "sext", "trunc",
    "select",
];

fn integer_definitions<'a>(lines: &[&'a str]) -> BTreeMap<&'a str, IntegerDefinition<'a>> {
    let mut compared_to_literal = BTreeSet::new();
    for line in lines {
        let Some((_, rhs)) = assignment(line) else {
            continue;
        };
        if !rhs.starts_with("icmp ") {
            continue;
        }
        let has_literal = rhs
            .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
            .any(|token| !token.is_empty() && token.parse::<i64>().is_ok());
        if has_literal {
            compared_to_literal.extend(value_tokens(rhs));
        }
    }

    let mut definitions = BTreeMap::new();
    for line in lines {
        let Some((lhs, rhs)) = assignment(line) else {
            continue;
        };
        let Some(opcode) = rhs.split_whitespace().next() else {
            continue;
        };
        if !INDEX_ARITHMETIC.contains(&opcode) {
            continue;
        }

        let has_literal_operand = rhs
            .split(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '[' | ']'))
            .any(|token| !token.is_empty() && token.parse::<i64>().is_ok());
        definitions.insert(
            lhs,
            IntegerDefinition {
                opcode,
                operands: value_tokens(rhs).collect(),
                has_literal_operand,
                is_bool_zext: rhs.starts_with("zext i1 "),
                compared_to_literal: compared_to_literal.contains(lhs),
            },
        );
    }
    definitions
}

/// Whether `index` is computed from a loop induction variable: its operand
/// graph reaches a cycle made only of `phi`s and steps by a literal (`%i.next =
/// add %i, 1`) or by a widened `i1` (Rust's range desugaring). Unrolling that
/// loop is what turns such an index into a constant. A recurrence that steps
/// by loaded data, or mixes in other arithmetic such as a hash's `mul`, is not
/// a counter, and unrolling would not help it. Nor is a counter whose loop
/// bound is only known at run time.
fn derives_from_induction_variable(
    index: &str,
    definitions: &BTreeMap<&str, IntegerDefinition<'_>>,
) -> bool {
    /// Depth-first search state per value: on the current path, or finished.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Visit {
        OnPath,
        Done,
    }

    fn visit<'a>(
        value: &'a str,
        definitions: &BTreeMap<&'a str, IntegerDefinition<'a>>,
        state: &mut BTreeMap<&'a str, Visit>,
        path: &mut Vec<&'a str>,
    ) -> bool {
        match state.get(value) {
            Some(Visit::OnPath) => {
                // a cycle: an induction variable when it is made only of phis
                // and counter-like steps. Any other opcode in the cycle (the
                // `mul` of a `h * a + c` hash) makes the value data that
                // unrolling would not turn into a constant
                let start = path.iter().rposition(|step| *step == value).unwrap_or(0);
                let cycle = &path[start..];
                let is_step = |definition: &IntegerDefinition<'_>| {
                    matches!(definition.opcode, "add" | "sub" | "or")
                        && (definition.has_literal_operand
                            || definition.operands.iter().any(|operand| {
                                definitions
                                    .get(operand)
                                    .is_some_and(|operand| operand.is_bool_zext)
                            }))
                };
                let only_counter_ops = cycle.iter().all(|step| {
                    let definition = &definitions[step];
                    definition.opcode == "phi" || is_step(definition)
                });
                let has_phi = cycle.iter().any(|step| definitions[step].opcode == "phi");
                let steps = cycle.iter().any(|step| is_step(&definitions[step]));
                // a runtime trip count (`0..n`) keeps the index dynamic even
                // under `#[unroll]`, so only a literal bound qualifies
                let bounded = cycle
                    .iter()
                    .any(|step| definitions[step].compared_to_literal);
                return only_counter_ops && has_phi && steps && bounded;
            }
            Some(Visit::Done) => return false,
            None => {}
        }
        let Some(definition) = definitions.get(value) else {
            return false;
        };

        state.insert(value, Visit::OnPath);
        path.push(value);
        let found = definition
            .operands
            .iter()
            .any(|operand| visit(operand, definitions, state, path));
        path.pop();
        state.insert(value, Visit::Done);
        found
    }

    visit(index, definitions, &mut BTreeMap::new(), &mut Vec::new())
}

fn address_escapes(line: &str, pointers: &BTreeSet<String>) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with(';') || trimmed.contains(" = getelementptr ") {
        return false;
    }

    if (trimmed.starts_with("call ") || trimmed.contains(" = call "))
        && !trimmed.contains("@llvm.")
        && pointers
            .iter()
            .any(|value| contains_value_token(trimmed, value))
    {
        return true;
    }

    if trimmed.starts_with("ret ")
        && pointers
            .iter()
            .any(|value| contains_value_token(trimmed, value))
    {
        return true;
    }

    if trimmed.contains(" = ptrtoint ")
        && pointers
            .iter()
            .any(|value| contains_value_token(trimmed, value))
    {
        return true;
    }

    if let Some(store) = trimmed.strip_prefix("store ")
        && let Some(stored_value) = store.split(',').next()
        && pointers
            .iter()
            .any(|value| contains_value_token(stored_value, value))
    {
        return true;
    }

    false
}

fn contains_value_token(text: &str, value: &str) -> bool {
    let mut search_from = 0usize;
    while let Some(relative) = text[search_from..].find(value) {
        let start = search_from + relative;
        let end = start + value.len();
        let before_ok = text[..start]
            .chars()
            .next_back()
            .is_none_or(|ch| !llvm_name_char(ch));
        let after_ok = text[end..]
            .chars()
            .next()
            .is_none_or(|ch| !llvm_name_char(ch));
        if before_ok && after_ok {
            return true;
        }
        search_from = start + 1;
    }
    false
}

fn llvm_name_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-')
}

fn value_tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|ch: char| ch.is_ascii_whitespace() || matches!(ch, ',' | '(' | ')' | '[' | ']'))
        .filter(|token| token.starts_with('%'))
}

fn format_diagnostic(
    function: &str,
    provenance: &LocalProvenance,
    blocker: PromotionBlocker,
) -> String {
    let binding = if provenance.name.is_empty() {
        format!("_{}", provenance.local_index)
    } else {
        provenance.name.clone()
    };
    let bytes = if provenance.size_bytes == 1 {
        "1 byte".to_string()
    } else {
        format!("{} bytes", provenance.size_bytes)
    };
    let mut message = format!(
        "warning: local `{binding}` (`{}`, {bytes}) in `{function}` could not be promoted to registers",
        provenance.type_name
    );

    match blocker {
        PromotionBlocker::LoopCounterIndex => {
            message.push_str(
                "\n  = note: indexed by the counter of a loop LLVM did not unroll, so the array stays in local memory",
            );
            message.push_str(
                "\n  = help: add `#[unroll]` to that loop so every index becomes a constant; if it already has one, raise its budget with `#[unroll(max_ops = N)]`",
            );
        }
        PromotionBlocker::DynamicIndex => {
            message.push_str(
                "\n  = note: indexed by a non-constant value; the allocation survives LLVM scalar replacement and remains stack-backed",
            );
            message.push_str(
                "\n  = help: use constant indexing or a small match/select when the index range is bounded",
            );
        }
        PromotionBlocker::AddressEscapes => {
            message.push_str(
                "\n  = note: the allocation's address escapes as a first-class pointer, preventing scalar replacement",
            );
            message.push_str(
                "\n  = help: keep accesses local to the value when possible so LLVM can promote the storage",
            );
        }
        PromotionBlocker::SurvivedScalarReplacement => {
            message.push_str(
                "\n  = note: the allocation survives mem2reg and LLVM scalar replacement and remains stack-backed",
            );
            message.push_str(
                "\n  = help: inspect dynamic indexing, pointer escapes, volatile accesses, or aliasing around this local; large aggregates are also harder to promote because scalar replacement must split every access into independent scalars",
            );
        }
    }

    message
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(provenance: &str) -> String {
        let mut output = String::new();
        for byte in provenance.as_bytes() {
            use std::fmt::Write as _;
            write!(&mut output, "{byte:02x}").unwrap();
        }
        output.push('_');
        output
    }

    fn alloca_name(provenance: &str) -> String {
        format!("{LOCAL_MEMORY_VALUE_PREFIX}{}", encoded(provenance))
    }

    #[test]
    fn dynamic_index_reports_the_rust_local() {
        let name = alloca_name("3\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(i64 %idx) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr {name}, i64 0, i64 %idx\n  %value = load i32, ptr %elt, align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `scratch` (`[u32; 4]`, 16 bytes)"));
        assert!(diagnostics[0].contains("indexed by a non-constant value"));
    }

    /// The rolled `for e in 0..4 { r[e] = .. }` fill, as `opt` leaves it
    fn rolled_counter_loop(name: &str) -> String {
        format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x float], align 4\n  br label %body\n\nbody:\n  %next = phi i64 [ 1, %entry ], [ %step, %body ]\n  %e = phi i64 [ 0, %entry ], [ %next, %body ]\n  %slot = getelementptr inbounds nuw [4 x i8], ptr {name}, i64 %e\n  store float 1.0, ptr %slot, align 4\n  %step = add nuw nsw i64 %next, 1\n  %done = icmp eq i64 %step, 5\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        )
    }

    #[test]
    fn rolled_loop_counter_index_is_reported_by_default() {
        let name = alloca_name("4\t16\tr\t[f32; 4]");
        let diagnostics =
            diagnose_text(&rolled_counter_loop(&name), Reporting::LoopCounterIndexOnly);
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        assert!(diagnostics[0].contains("local `r` (`[f32; 4]`, 16 bytes)"));
        assert!(diagnostics[0].contains("counter of a loop"));
        assert!(diagnostics[0].contains("#[unroll]"));
    }

    #[test]
    fn range_next_counter_steps_by_a_widened_bool() {
        // `for k in 0..8` as opt leaves it: `Range::next` adds `zext i1`
        let name = alloca_name("30\t128\tv\t[F32x4; 8]");
        let llvm = format!(
            "define void @softmax() {{\nentry:\n  {name} = alloca [8 x {{ [4 x float] }}], align 16\n  br label %head\n\nhead:\n  %next = phi i64 [ 1, %entry ], [ %step, %latch ]\n  %k = phi i64 [ 0, %entry ], [ %next, %latch ]\n  %slot = getelementptr inbounds nuw [16 x i8], ptr {name}, i64 %k\n  store float 1.0, ptr %slot, align 16\n  br label %latch\n\nlatch:\n  %more = icmp ult i64 %next, 8\n  %inc = zext i1 %more to i64\n  %step = add nuw nsw i64 %next, %inc\n  br i1 %more, label %head, label %exit\n\nexit:\n  ret void\n}}\n"
        );
        assert_eq!(
            diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).len(),
            1
        );
    }

    #[test]
    fn data_dependent_recurrence_is_not_a_loop_counter() {
        // an index that steps by loaded data stays dynamic after unrolling
        let name = alloca_name("3\t16\tring\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(ptr %deltas) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  br label %body\n\nbody:\n  %pos = phi i64 [ 0, %entry ], [ %pos.next, %body ]\n  %slot.index = and i64 %pos, 3\n  %slot = getelementptr [4 x i32], ptr {name}, i64 0, i64 %slot.index\n  store i32 1, ptr %slot, align 4\n  %delta = load i64, ptr %deltas, align 8\n  %pos.next = add i64 %pos, %delta\n  %done = icmp eq i64 %pos.next, 64\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
    }

    #[test]
    fn runtime_trip_count_is_not_reported_by_default() {
        // `for i in 0..n` with `n` a kernel argument: `#[unroll]` cannot make
        // `arr[i]` constant, so the default build stays quiet
        let name = alloca_name("3\t16\tarr\t[u32; 4]");
        let llvm = rolled_counter_loop(&name)
            .replace("@kernel()", "@kernel(i64 %n)")
            .replace("icmp eq i64 %step, 5", "icmp eq i64 %step, %n");
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn hash_recurrence_is_not_a_loop_counter() {
        // `h = h * a + c` steps by a literal but is data: its `mul` keeps every
        // unrolled index dynamic
        let name = alloca_name("3\t16\tt\t[f32; 256]");
        let llvm = format!(
            "define void @kernel(i32 %seed) {{\nentry:\n  {name} = alloca [256 x float], align 4\n  br label %body\n\nbody:\n  %h = phi i32 [ %seed, %entry ], [ %h.next, %body ]\n  %n = phi i32 [ 0, %entry ], [ %n.next, %body ]\n  %scaled = mul i32 %h, 1664525\n  %h.next = add i32 %scaled, 1013904223\n  %top = lshr i32 %h.next, 24\n  %slot.index = zext i32 %top to i64\n  %slot = getelementptr [256 x float], ptr {name}, i64 0, i64 %slot.index\n  %value = load float, ptr %slot, align 4\n  %n.next = add i32 %n, 1\n  %done = icmp eq i32 %n.next, 64\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
    }

    #[test]
    fn runtime_index_is_reported_only_in_verbose_builds() {
        let name = alloca_name("3\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(i64 %idx) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr {name}, i64 0, i64 %idx\n  %value = load i32, ptr %elt, align 4\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        assert_eq!(diagnose_text(&llvm, Reporting::All).len(), 1);
    }

    #[test]
    fn index_loaded_inside_a_loop_is_not_a_loop_counter() {
        // a data-dependent index in a loop: unrolling would not make it constant
        let name = alloca_name("3\t16\thist\t[u32; 4]");
        let llvm = format!(
            "define void @kernel(ptr %keys) {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  br label %body\n\nbody:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %body ]\n  %key.ptr = getelementptr i64, ptr %keys, i64 %i\n  %key = load i64, ptr %key.ptr, align 8\n  %bin = getelementptr [4 x i32], ptr {name}, i64 0, i64 %key\n  store i32 1, ptr %bin, align 4\n  %i.next = add i64 %i, 1\n  %done = icmp eq i64 %i.next, 64\n  br i1 %done, label %exit, label %body\n\nexit:\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::LoopCounterIndexOnly).is_empty());
        let verbose = diagnose_text(&llvm, Reporting::All);
        assert!(
            verbose[0].contains("indexed by a non-constant value"),
            "{verbose:?}"
        );
    }

    #[test]
    fn optimizer_suffixes_do_not_hide_surviving_allocas() {
        let provenance = "8\t64\tvalues\t[u64; 8]";
        let name = format!("{}.sroa.0", alloca_name(provenance));
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [8 x i64], align 8\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `values`"));
    }

    #[test]
    fn name_uniquing_suffixes_do_not_garble_the_report() {
        // LLVM uniques colliding value names by appending bare digits: the
        // same local inlined twice into one kernel yields `<name>` and
        // `<name>1`. The `_` sentinel ends the hex payload, so the digit must
        // not leak into the decoded fields.
        let provenance = "3\t16\tscratch\t[u32; 4]";
        let base = alloca_name(provenance);
        let uniqued = format!("{base}1");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {base} = alloca [4 x i32], align 4\n  {uniqued} = alloca [4 x i32], align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 2);
        for diagnostic in &diagnostics {
            assert!(
                diagnostic.contains("local `scratch` (`[u32; 4]`, 16 bytes)"),
                "uniquing digit corrupted the decoded provenance: {diagnostic}"
            );
        }
    }

    #[test]
    fn capped_payloads_still_attribute_the_local() {
        // The exporter caps the payload length before hex-encoding; a
        // truncated trailing type spelling must not lose the report.
        let full = format!("7\t4096\tstate\t{}", "VeryLongTypeName".repeat(8));
        let truncated = &full[..40];
        let name = alloca_name(truncated);
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [512 x i64], align 8\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("local `state`"));
        assert!(diagnostics[0].contains("4096 bytes"));
    }

    #[test]
    fn payload_without_the_sentinel_is_not_trusted() {
        // Without the terminating `_` a trailing uniquing digit is
        // indistinguishable from payload, so such names must be ignored
        // rather than half-decoded.
        let provenance = "3\t16\tscratch\t[u32; 4]";
        let name = alloca_name(provenance);
        let unterminated = name.trim_end_matches('_');
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {unterminated} = alloca [4 x i32], align 4\n  ret void\n}}\n"
        );
        assert!(diagnose_text(&llvm, Reporting::All).is_empty());
    }

    #[test]
    fn address_escape_gets_a_specific_reason() {
        let name = alloca_name("4\t32\tstate\tState");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca %State, align 8\n  call void @consume(ptr {name})\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("address escapes"));
    }

    #[test]
    fn llvm_lifetime_intrinsics_are_not_reported_as_address_escapes() {
        let name = alloca_name("5\t16\tscratch\t[u32; 4]");
        let llvm = format!(
            "define void @kernel() {{\nentry:\n  {name} = alloca [4 x i32], align 4\n  call void @llvm.lifetime.start.p0(i64 16, ptr {name})\n  call void @llvm.lifetime.end.p0(i64 16, ptr {name})\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("survives mem2reg and LLVM scalar replacement"));
        assert!(!diagnostics[0].contains("address escapes"));
    }

    #[test]
    fn mangled_containing_functions_are_demangled() {
        let name = alloca_name("2\t8\tacc\t[f32; 2]");
        let llvm = format!(
            "define void @_ZN6kernel5inner17h0123456789abcdefE() {{\nentry:\n  {name} = alloca [2 x float], align 4\n  ret void\n}}\n"
        );
        let diagnostics = diagnose_text(&llvm, Reporting::All);
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].contains("in `kernel::inner`"),
            "expected demangled function name without hash: {}",
            diagnostics[0]
        );
    }

    #[test]
    fn untagged_codegen_allocas_are_ignored() {
        let llvm = "define void @kernel(i64 %idx) {\nentry:\n  %tmp = alloca [4 x i32], align 4\n  %elt = getelementptr [4 x i32], ptr %tmp, i64 0, i64 %idx\n  ret void\n}\n";
        assert!(diagnose_text(llvm, Reporting::All).is_empty());
    }

    #[test]
    fn promoted_local_is_silent_when_the_alloca_is_absent() {
        let llvm = "define i32 @kernel(i32 %value) {\nentry:\n  ret i32 %value\n}\n";
        assert!(diagnose_text(llvm, Reporting::All).is_empty());
    }
}
