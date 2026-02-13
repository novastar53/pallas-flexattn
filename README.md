# Pallas FlexAttention

A FlexAttention implementation in JAX/Pallas, inspired by PyTorch's FlexAttention.

## Overview

This repository contains a from-scratch implementation of Flash Attention with FlexAttention features:
- Custom score modification functions
- Block-sparse attention masks
- Autotuning for optimal kernel configurations
- Support for causal, sliding window, and arbitrary attention patterns

