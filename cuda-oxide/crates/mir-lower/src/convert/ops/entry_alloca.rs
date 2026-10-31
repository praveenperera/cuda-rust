/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fixed-size stack slots placed in the function's entry block.
//!
//! LLVM only treats an `alloca` as static when it sits in the entry block
//! with a constant count. Anywhere else it is a dynamic allocation: SROA and
//! mem2reg leave it alone, the inliner refuses ordinary cost-driven inlining
//! (`has uninlinable pattern (dynamic alloca)`), and forced inlining brackets
//! it with `llvm.stacksave` / `llvm.stackrestore`. The slot then survives as a
//! `.local` frame in PTX on both the `llc` and the libNVVM routes (#1687,
//! #1681, #1590).
//!
//! Every slot mir-lower creates is one element of a fixed type, so it is
//! emitted here, at the end of the entry block, while the stores and loads
//! that use it stay at the use site.
//!
//! # When one slot per op is enough
//!
//! A hoisted slot is one address that every execution of its op reuses,
//! the model rustc's own LLVM backend uses for every MIR local (all of its
//! allocas go to the entry block). That is sound when no earlier
//! execution's address is still in use when the op runs again:
//!
//! - Pure scratch (transmute through memory, dynamic array indexing, enum
//!   payload spills, libdevice out-parameters) is written, then read, inside
//!   the lowering of one op; the address never leaves that op.
//! - A `mir.alloca` already belongs to the MIR entry block and runs once per
//!   call; moving it ahead of the prologue branch changes nothing else.
//!
//! `mir.ref` of an SSA value is different: it returns the slot's address,
//! and that reference can outlive the op. The importer takes `mir.ref` of
//! temporaries it synthesizes itself (for example a copied enum payload),
//! which borrowck never sees, so a loop can carry the reference from one
//! iteration into the next while the same op stores a new value. With one
//! hoisted slot both references would read the newer value. So
//! [`emit_ref_slot`] hoists only when the op's block lies on no CFG cycle;
//! such a block runs at most once per call, and one slot is then exactly
//! one execution. Inside a loop it keeps a use-site alloca, which LLVM
//! allocates afresh on every execution.

use crate::helpers;
use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::{Inserter, OpInsertionPoint};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use rustc_hash::FxHashSet;
use std::num::NonZeroUsize;

/// Emit `alloca <ty>, i32 1` in the entry block of the function that holds
/// the rewriter's current insertion point, and return the alloca operation.
///
/// The caller's insertion point stays at the use site, so initialization
/// and accesses emitted next land there. `align` sets the slot's
/// alignment when given. Callers that replace a MIR op with the slot can pass
/// the returned operation to `replace_operation`; others read result 0.
pub(crate) fn emit_entry_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ty: TypeHandle,
    align: Option<u32>,
) -> Result<Ptr<Operation>> {
    let use_point = rewriter.get_insertion_point();
    let use_block = rewriter.get_insertion_block(ctx).ok_or_else(|| {
        pliron::input_error_noloc!("stack slot requested with no insertion block")
    })?;
    let function = helpers::get_parent_func(ctx, use_block)
        .map_err(|error| pliron::input_error_noloc!("{error}"))?;
    let entry = function
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| pliron::input_error_noloc!("stack slot requested in a bodiless function"))?;

    // the entry block holds the prologue and a branch to the first MIR block;
    // slots go before that branch so they keep creation order, and the count
    // constant goes with them so it dominates the alloca. A use already in
    // the entry block keeps its own insertion point, which then advances
    // past the slot like any other emitted op
    let hoist = use_block != entry;
    if hoist {
        rewriter.set_insertion_point(match entry.deref(ctx).get_terminator(ctx) {
            Some(terminator) => OpInsertionPoint::BeforeOperation(terminator),
            None => OpInsertionPoint::AtBlockEnd(entry),
        });
    }

    let alloca = insert_alloca(ctx, rewriter, ty, align);

    if hoist {
        rewriter.set_insertion_point(use_point);
    }
    Ok(alloca)
}

/// Emit the slot that `mir.ref` of an SSA value stores into, and return the
/// alloca operation.
///
/// The slot goes to the entry block, as with [`emit_entry_alloca`], unless
/// the use site's block lies on a CFG cycle. There a reference from an
/// earlier execution may still be live, so the slot stays at the use site
/// and each execution gets its own storage (see the module docs).
pub(crate) fn emit_ref_slot(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ty: TypeHandle,
    align: Option<u32>,
) -> Result<Ptr<Operation>> {
    let use_block = rewriter.get_insertion_block(ctx).ok_or_else(|| {
        pliron::input_error_noloc!("stack slot requested with no insertion block")
    })?;
    if block_on_cycle(ctx, use_block) {
        Ok(insert_alloca(ctx, rewriter, ty, align))
    } else {
        emit_entry_alloca(ctx, rewriter, ty, align)
    }
}

/// Insert `alloca <ty>, i32 1` and its count constant at the rewriter's
/// current insertion point.
fn insert_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ty: TypeHandle,
    align: Option<u32>,
) -> Ptr<Operation> {
    let one = IntegerAttr::new(
        IntegerType::get(ctx, 32, Signedness::Signless),
        APInt::from_u64(1, NonZeroUsize::new(32).unwrap()),
    );
    let one = llvm::ConstantOp::new(ctx, Box::new(one));
    rewriter.insert_operation(ctx, one.get_operation());
    let one = one.get_operation().deref(ctx).get_result(0);
    let alloca = llvm::AllocaOp::new(ctx, ty, one, 0).get_operation();
    if let Some(align) = align {
        llvm::set_op_alignment(ctx, alloca, align);
    }
    rewriter.insert_operation(ctx, alloca);
    alloca
}

/// Whether `block` can reach itself through the function's CFG.
///
/// Conversion is mid-flight, so a block's terminator may be a MIR or an
/// LLVM op; both expose their successors. A reachable block whose last op
/// is not a terminator has unknown successors, and is treated as closing a
/// cycle so the caller keeps the per-execution slot.
fn block_on_cycle(ctx: &Context, block: Ptr<BasicBlock>) -> bool {
    let mut visited = FxHashSet::default();
    let mut pending = vec![block];
    while let Some(current) = pending.pop() {
        let Some(terminator) = current.deref(ctx).get_terminator(ctx) else {
            return true;
        };
        for successor in terminator.deref(ctx).successors() {
            if successor == block {
                return true;
            }
            if visited.insert(successor) {
                pending.push(successor);
            }
        }
    }
    false
}
