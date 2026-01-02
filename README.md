# neural-network-typescript

A TypeScript implementation of a feedforward neural network with support for various activation functions and gradient descent optimization.

## Features

- Multiple activation functions (Sigmoid, ReLU, Leaky ReLU, Linear, Tanh, SoftPlus)
- Configurable network architecture
- Forward propagation with customizable activations
- Gradient computation via backpropagation
- Batch gradient descent optimization
- Type-safe implementation

## Installation

```typescript
import { Network, sigmoid, reLU } from "./neural-network";
```

## Architecture

The network is constructed by specifying layer sizes and activation functions:

```typescript
const network = new Network(
  [inputSize, linear], // Input layer
  [hiddenSize, reLU], // Hidden layer
  [outputSize, sigmoid] // Output layer
);
```

## Activation Functions

All activation functions implement both the activation and its derivative:

- `sigmoid`: $\sigma(x) = 1 / (1 + e^{-x}) \Rightarrow \sigma'(x) = \sigma(x) \times (1 - \sigma(x))$
- `reLU`: $\text{ReLU}(x) = \max(0, x) \Rightarrow \text{ReLU}'(x) = \begin{cases} 1, \text{for } x \geq 0 \\ 0, \text{for } x < 0 \end{cases}$
- `leakyReLU(alpha)`: $\text{Leaky ReLU}(x) = \max(x, \alpha \times x) \Rightarrow \text{Leaky ReLU}'(x) = \begin{cases} 1, \text{for } x \geq 0 \\ \alpha, \text{for } x < 0 \end{cases}$ where $0 < \alpha < 1$
- `linear`: $\lambda(x) = x \Rightarrow \lambda'(x) = 1$
- `tanh`: $\tanh(x) = 2 \times \sigma(2 \times x) - 1 \Rightarrow \tanh'(x) = 4 \times \sigma'(2 \times x)$
- `softPlus`: $\text{Soft}_{+}(x) = \ln(1 + e^x) \Rightarrow \text{Soft}_{+}'(x) = \sigma(x)$

## Core Algorithms

### Forward Propagation (`getOutput`)

Given input vector $x$, the network computes activations layer by layer:

For layer $(k)$ ($1 \leq k \leq L$):

$$
z^{(k)}_i = \sum_i (w^{(k)}_{ij} * a^{(k-1)}_j) + b^{(k)}_i
$$

$$
a^{(k)}_i = \text{activation}^{(k)}(z^{(k)}_i)
$$

Where:

- $z^{(k)}_i$ is the weighted input to neuron $i$ in layer $(k)$
- $w^{(k)}_{ij}$ is the weight from neuron $j$ in layer $(k-1)$ to neuron $i$ in layer $(k)$
- $a^{(k-1)}_j$ is the activation of neuron $j$ in previous layer
- $b^{(k)}_i$ is the bias for neuron $i$ in layer $(k)$
- $\text{activation}^{(k)}(x)$ is the activation function for layer $(k)$

The input layer $a^{(0)}$ is simply the input vector $x$.

### Gradient Computation (`getGradient`)

Using backpropagation, the algorithm computes gradients for the cost function:

$$
C = \sum_i (a^{(L)}_i - y_i)^2
$$

1. **Forward pass**: Compute all activations $a^{(k)}_i$ and weighted inputs $z^{(k)}_i$
2. **Output layer error**:
   $$
   \frac{dC}{da^{(L)}_i} = 2 \times (a^{(L)}_i - y_i)
   $$
3. **Backpropagate error** (for $k = L-1 \text{ to } 1$):
   $$
   \delta^{(k)}_i = \frac{dC}{da^{(k)}_i} \times \text{activation}'^{(k)}(z^{(k)}_i)
   $$
   $$
   \frac{dC}{da^{(k - 1)}_j} = \sum_i \delta^{(k)}_i w^{(k)}_{ij}
   $$
4. **Compute gradients**:
   $$
   \nabla C (w^{(k)}_{ij}) = \delta^{(k)}_i \times a^{(k - 1)}_j
   $$
   $$
   \nabla C (b^{(k)}_i) = \delta^{(k)}_i
   $$

## API Reference

### Network Class

#### Constructor

```typescript
new Network(...layers: [number, Activation][])
```

#### Methods

- `getOutput(input: number[]): number[][]` - Forward propagation
- `getCost(input: number[], target: number[]): number` - Mean squared error
- `getGradient(input: number[], target: number[]): Gradient` - Single example gradients
- `getMeanGradient(inputs: number[][], targets: number[][]): Gradient` - Batch gradients
- `gradientDescend(batchInputs: number[][][], batchTargets: number[][][])` - Perform gradient descent

### Types

```typescript
interface Gradient {
  nabla_w: number[][][]; // Weight gradients
  nabla_b: number[][]; // Bias gradients
}

interface Activation {
  activation: (x: number) => number;
  dActivation: (x: number) => number;
}
```

## Example Usage

```typescript
// Create a network with 2 input, 3 hidden, and 1 output neurons
const network = new Network([2, linear], [3, reLU], [1, sigmoid]);

// Forward pass
const input = [0.5, -0.3];
const output = network.getOutput(input);
console.log(output[output.length - 1]); // Final layer output

// Compute cost and gradients
const target = [0.8];
const cost = network.getCost(input, target);
const gradient = network.getGradient(input, target);

// Training with batch gradient descent
const trainingData = [[[0.5, -0.3]], [[0.2, 0.4]]];
const targets = [[[0.8]], [[0.6]]];
network.gradientDescend(trainingData, targets);
```

## Notes

- Weights are initialized randomly between -1 and 1
- The first layer uses linear activation (no transformation)
- The cost function is mean squared error (MSE)
- Gradient descent updates weights and biases immediately

## Error Handling

The library includes validation for:

- Input/layer size mismatches
- Invalid alpha values for leaky ReLU
- Batch size consistency
- Layer index bounds
