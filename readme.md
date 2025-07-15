# Federated Very Fast Decision Tree (Fed-VFDT)

This repository provides a prototype implementation of a **federated decision tree learning system** based on the **Very Fast Decision Tree (VFDT)** algorithm. The system is designed for **streaming, distributed, and potentially unlabeled data**, with global coordination across clients using federated aggregation strategies.

The implementation combines [Flower](https://flower.dev/) for federated orchestration and [River](https://riverml.xyz/) for data stream learning. Each client trains a local VFDT model and periodically shares split proposals with the server. The server aggregates these proposals and enforces a **synchronized global split** decision, ensuring model consistency across clients.

## Features

- Federated adaptation of the VFDT algorithm (Hoeffding Trees)
- Client-side training with local split proposals
- Server-side coordination of tree growth via aggregation
- Global synchronization of splits using feature-level consensus
- Support for binary and multinomial splits
- Communication-efficient: only sends messages on split attempts
- Designed for online and incremental learning on streaming data

## Requirements

- Python 3.8+
- [`flower`](https://flower.dev/)
- [`river`](https://riverml.xyz/)
- NumPy, pandas, gRPC, etc.

