# Tilecute

Tilecute is an experimental backend implementation for Tilelang using cuteDSL. This project demonstrates the capabilities of cuteDSL through a direct 1:1 mapping approach to Tilelang's semantics.

## Purpose

- Direct 1:1 mapping from Tilelang to cuteDSL representations
- Experimental backend implementation
- Demonstration of cuteDSL's expressive power and flexibility

## Current Status

[x] elementwise: vectorized LDG & STG, elementwise opertation, custom layout.
[ ] gemm on Ampere: LDGSTS, Tensor core, pipeline.
[ ] gemm on Hopper: TMA, warp specialization, mbarrier operation, WGMMA.

## Getting Started

pip install tilelang, nvidia-cutlass-dsl


