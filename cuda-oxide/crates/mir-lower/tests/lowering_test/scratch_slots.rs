/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops as mir;
use dialect_mir::types::{MirArrayType, MirPointerKind, MirPtrType};
use llvm_export::export::{NvvmExportConfig, NvvmIrDialect};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, TypedHandle};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

use super::common::{append_return, build_test_kernel, make_test_ctx};

/// Each `alloca` line of the function, paired with its block label.
fn allocas_by_block(ir: &str) -> Vec<(&str, &str)> {
    let mut block = "";
    let mut allocas = Vec::new();
    for line in ir.lines() {
        if line.starts_with("define ") {
            block = "";
        } else if let Some(label) = line.strip_suffix(':')
            && !label.starts_with(' ')
        {
            block = label;
        } else if line.contains(" = alloca ") {
            allocas.push((block, line.trim()));
        }
    }
    allocas
}

/// The instruction lines of the block labelled `label`, trimmed.
fn block_lines<'a>(ir: &'a str, label: &'a str) -> impl Iterator<Item = &'a str> {
    ir.lines()
        .skip_while(move |line| line.strip_suffix(':') != Some(label))
        .skip(1)
        .take_while(|line| line.starts_with(' '))
        .map(str::trim)
}

/// Compiler scratch for ops outside the entry block -- `mir.ref` of an SSA
/// value, a dynamic array index and a transmute through memory -- is emitted
/// as an entry-block alloca, with its store left at the use site. LLVM treats
/// any other alloca as dynamic: SROA skips it, ordinary inlining refuses the
/// function and forced inlining adds `stacksave`/`stackrestore`, leaving a
/// `.local` frame on both the `llc` and the libNVVM routes (#1687, #1681)
#[test]
fn scratch_slots_are_entry_block_allocas() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![u32_ty]);
    let index = entry.deref(&ctx).get_argument(0);
    let region = entry
        .deref(&ctx)
        .get_parent_region()
        .expect("entry in a region");
    let body = BasicBlock::new(&mut ctx, None, vec![]);
    body.insert_at_back(region, &ctx);
    Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![body],
        0,
    )
    .insert_at_back(entry, &ctx);

    let ptr_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::SharedRef);
    let reference = Operation::new(
        &mut ctx,
        mir::MirRefOp::get_concrete_op_info(),
        vec![ptr_ty.into()],
        vec![index],
        vec![],
        0,
    );
    let mir_ref = mir::MirRefOp::new(reference);
    mir_ref.set_mutable(&mut ctx, false);
    mir_ref.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    reference.insert_at_back(body, &ctx);

    let array_ty: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 4).into();
    let array = mir::MirUndefOp::new(&mut ctx, array_ty).get_operation();
    array.insert_at_back(body, &ctx);
    let array = array.deref(&ctx).get_result(0);
    Operation::new(
        &mut ctx,
        mir::MirExtractArrayElementOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![array, index],
        vec![],
        0,
    )
    .insert_at_back(body, &ctx);

    let u128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Unsigned).into();
    let transmute = Operation::new(
        &mut ctx,
        mir::MirCastOp::get_concrete_op_info(),
        vec![u128_ty],
        vec![array],
        vec![],
        0,
    );
    mir::MirCastOp::new(transmute).set_attr_cast_kind(&ctx, MirCastKindAttr::Transmute);
    transmute.insert_at_back(body, &ctx);
    append_return(&mut ctx, body);

    let exports = lower_and_export(&mut ctx, module_ptr)?;
    for (route, ir) in &exports {
        let allocas = allocas_by_block(ir);
        assert_eq!(allocas.len(), 3, "{route}: one slot per scratch op:\n{ir}");
        for (block, alloca) in allocas {
            assert_eq!(block, "entry", "{route}: `{alloca}` outside entry:\n{ir}");
        }
        // initialization stays where the value is defined, not in entry
        let body = ir
            .split_once("\nbb0:")
            .map(|(_, rest)| rest)
            .ok_or_else(|| anyhow::anyhow!("{route}: no body block:\n{ir}"))?;
        assert_eq!(body.matches("  store ").count(), 3, "{route}:\n{ir}");
    }
    Ok(())
}

/// A `mir.ref` in a loop must give every execution its own storage: the
/// reference from one iteration can still be live when the op runs again.
/// The kernel keeps `&(0 + 11)` from the first iteration and `&(1 + 11)`
/// from the second and stores `*second - *first`, which is 1 only when the
/// two references name different slots. Hoisting that slot to the entry
/// block made both read 12. The `mir.ref` outside the loop is still hoisted
#[test]
fn loop_ref_slots_are_per_execution() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
    let ref_ty: TypeHandle = MirPtrType::get_generic_with_kind(
        &mut ctx,
        u32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    )
    .into();
    let out_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true).into();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![out_ty]);
    let out = entry.deref(&ctx).get_argument(0);
    let region = entry
        .deref(&ctx)
        .get_parent_region()
        .expect("entry in a region");
    let header = BasicBlock::new(&mut ctx, None, vec![u32_ty.into(), ref_ty]);
    header.insert_at_back(region, &ctx);
    let exit = BasicBlock::new(&mut ctx, None, vec![ref_ty, ref_ty]);
    exit.insert_at_back(region, &ctx);

    // entry: goto header(0, &0)
    let zero = append_u32(&mut ctx, entry, u32_ty, 0);
    let initial = append_ref(&mut ctx, entry, ref_ty, zero);
    append_void::<mir::MirGotoOp>(&mut ctx, entry, vec![zero, initial], vec![header]);

    // header(i, prev): current = &(i + 11); i == 0 ? header(i + 1, current)
    // : exit(prev, current)
    let i = header.deref(&ctx).get_argument(0);
    let prev = header.deref(&ctx).get_argument(1);
    let eleven = append_u32(&mut ctx, header, u32_ty, 11);
    let value = append_value::<mir::MirAddOp>(&mut ctx, header, u32_ty.into(), vec![i, eleven]);
    let current = append_ref(&mut ctx, header, ref_ty, value);
    let one = append_u32(&mut ctx, header, u32_ty, 1);
    let next = append_value::<mir::MirAddOp>(&mut ctx, header, u32_ty.into(), vec![i, one]);
    let first = append_value::<mir::MirEqOp>(&mut ctx, header, bool_ty, vec![i, zero]);
    let (operands, segment_sizes) = mir::MirCondBranchOp::compute_segment_sizes(vec![
        vec![first],
        vec![next, current],
        vec![prev, current],
    ]);
    let branch = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![header, exit],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    branch.insert_at_back(header, &ctx);

    // exit(a, b): *out = *b - *a
    let a = exit.deref(&ctx).get_argument(0);
    let b = exit.deref(&ctx).get_argument(1);
    let a = append_value::<mir::MirLoadOp>(&mut ctx, exit, u32_ty.into(), vec![a]);
    let b = append_value::<mir::MirLoadOp>(&mut ctx, exit, u32_ty.into(), vec![b]);
    let difference = append_value::<mir::MirSubOp>(&mut ctx, exit, u32_ty.into(), vec![b, a]);
    append_void::<mir::MirStoreOp>(&mut ctx, exit, vec![out, difference], vec![]);
    append_return(&mut ctx, exit);

    let exports = lower_and_export(&mut ctx, module_ptr)?;
    for (route, ir) in &exports {
        let allocas = allocas_by_block(ir);
        assert_eq!(allocas.len(), 2, "{route}: one slot per mir.ref:\n{ir}");
        assert_eq!(
            allocas[0].0, "entry",
            "{route}: ref outside the loop:\n{ir}"
        );
        // the loop's slot must be allocated by the block that branches back
        // to itself, so each iteration gets fresh storage
        let (loop_block, alloca) = allocas[1];
        let self_edge = format!("label %{loop_block}");
        assert!(
            block_lines(ir, loop_block).any(|line| line.starts_with("br ")
                && line[3..].split(", ").any(|target| target == self_edge)),
            "{route}: `{alloca}` is not in the loop block:\n{ir}"
        );
    }
    Ok(())
}

/// Lower `module_ptr` and export it on the `llc`, legacy NVVM and modern
/// NVVM routes.
fn lower_and_export(
    ctx: &mut Context,
    module_ptr: Ptr<Operation>,
) -> Result<Vec<(&'static str, String)>, anyhow::Error> {
    mir_lower::lower_mir_to_llvm(ctx, module_ptr).map_err(|error| anyhow::anyhow!("{error}"))?;
    pliron::operation::verify_operation(module_ptr, ctx)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let module = Operation::get_op::<ModuleOp>(module_ptr, ctx).unwrap();
    let mut exports = vec![(
        "llc",
        llvm_export::export::export_module_to_string(ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?,
    )];
    for (route, dialect) in [
        ("legacy NVVM", NvvmIrDialect::LegacyLlvm7),
        ("modern NVVM", NvvmIrDialect::Modern),
    ] {
        let ir = llvm_export::export::export_module_to_string_with_config(
            ctx,
            &module,
            &NvvmExportConfig::new(dialect),
        )
        .map_err(|error| anyhow::anyhow!(error))?;
        exports.push((route, ir));
    }
    Ok(exports)
}

fn append_value<T: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    result: TypeHandle,
    operands: Vec<Value>,
) -> Value {
    let op = Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![result],
        operands,
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

fn append_void<T: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    operands: Vec<Value>,
    successors: Vec<Ptr<BasicBlock>>,
) {
    Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![],
        operands,
        successors,
        0,
    )
    .insert_at_back(block, ctx);
}

fn append_u32(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    ty: TypedHandle<IntegerType>,
    value: u32,
) -> Value {
    let op = Operation::new(
        ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(ty, APInt::from_u32(value, NonZeroUsize::new(32).unwrap())),
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// `&value` for an SSA value, which lowering backs with a stack slot.
fn append_ref(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle, value: Value) -> Value {
    let op = Operation::new(
        ctx,
        mir::MirRefOp::get_concrete_op_info(),
        vec![ty],
        vec![value],
        vec![],
        0,
    );
    let reference = mir::MirRefOp::new(op);
    reference.set_mutable(ctx, false);
    reference.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::Reborrow);
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}
