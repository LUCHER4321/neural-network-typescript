import { Activation, Gradient, NumericFunction } from "./types";

const random = (min: number = -1, max: number = 1) => Math.random() * (max - min) + min;

const outputToZ = (a: number[][], weights: number[][][], biases: number[][], layerIndex: number, neuronIndex: number) => {
    if (layerIndex <= 0) return 0;
    const z = weights[layerIndex][neuronIndex].reduce((sum, w, i) => sum + w * a[layerIndex - 1][i], 0) + biases[layerIndex][neuronIndex];
    return z;
};

export class Network {
    weights: number[][][] = [];
    biases: number[][] = [];
    activations: Activation[] = [];

    constructor(...layers: [number, Activation][]) {
        this.weights = layers.map(([layerSize], layerIndex) => {
            if (layerIndex === 0) return [];
            const [previousLayerSize] = layers[layerIndex - 1];
            return Array.from({ length: layerSize }, () =>
                Array.from({ length: previousLayerSize }, () => random())
            );
        });
        this.biases = layers.map(([layerSize], layerIndex) => {
            if (layerIndex === 0) return [];
            return Array.from({ length: layerSize }, () => random());
        });
        this.activations = layers.map(([_, activation]) => activation)
    }

    getOutput = (input: number[]) => {
        if(this.biases[0].length !== input.length) throw new Error(`Input and 1st layer must have the same length (${this.biases[0].length})`);
        const a = this.biases.map(layer => layer.map(() => 0));
        a[0] = input;
        for (let k = 1; k < this.biases.length; k++) {
            a[k] = this.biases[k].map((b, j) => {
                const z = this.weights[k][j].reduce((sum, w, i) => sum + w * a[k - 1][i], 0) + b;
                return this.activations[k].activation(z);
            });
        }
        return a
    }

    getCost = (input: number[], target: number[]) => {
        if(this.biases[0].length !== input.length) throw new Error(`Input and 1st layer must have the same length (${this.biases[0].length})`);
        const L = this.biases.length;
        if(this.biases[L - 1].length !== target.length) throw new Error(`Target and last layer must have the same length (${this.biases[L - 1].length})`);
        const a = this.getOutput(input);
        return a[L - 1].reduce((sum, output, i) => sum + Math.pow(output - target[i], 2))
    }

    getGradient = (input: number[], target: number[]): Gradient => {
        if(this.biases[0].length !== input.length) throw new Error(`Input and 1st layer must have the same length (${this.biases[0].length})`);
        const L = this.biases.length;
        if(this.biases[L - 1].length !== target.length) throw new Error(`Target and last layer must have the same length (${this.biases[L - 1].length})`);
        const a = this.getOutput(input);
        const dCda = this.biases.map(layer => layer.map(() => 0));
        const nabla_w = this.weights.map(layer => layer.map(neuron => neuron.map(() => 0)));
        const nabla_b = this.biases.map(layer => layer.map(() => 0));
        dCda[L - 1] = a[L - 1].map((output, i) => 2 * (output - target[i]));
        for (let k = L - 1; k >= 0; k--) {
            nabla_b[k] = dCda[k].map((dC, j) => dC * this.activations[k].dActivation(outputToZ(a, this.weights, this.biases, k, j)));
            nabla_w[k] = this.weights[k].map((weights, i) =>
                weights.map((_, j) => nabla_b[k][i] * a[k - 1][j])
            );
            if(k > 0) dCda[k - 1] = nabla_w[k].reduce((sum, nw) => sum.map((s, i) => s + nw[i] * nabla_b[k][i]));
        }
        return {
            nabla_w,
            nabla_b
        }
    }

    getMeanGradient = (input: number[][], target: number[][]): Gradient => {
        if(input.length !== target.length) throw new Error("Input and target must have the same length");
        const total = input.length;
        const gradients: Gradient[] = input.map((i, n) => this.getGradient(i, target[n]));
        const totalGradient = gradients.reduce((sum, { nabla_w, nabla_b }) => ({
            nabla_w: sum.nabla_w.map((s, k) => s.map((s1, i) => s1.map((s2, j) => s2 + nabla_w[k][i][j]))),
            nabla_b: sum.nabla_b.map((s, k) => s.map((s1, i) => s1 + nabla_b[k][i]))
        }));
        return {
            nabla_w: totalGradient.nabla_w.map(k => k.map(i => i.map(j => j / total))),
            nabla_b: totalGradient.nabla_b.map(k => k.map(i => i / total))
        }
    }

    gradientDescend = (input: number[][][], target: number[][][]) => {
        if(input.length !== target.length) throw new Error("Input and target must have the same length");
        const total = input.length;
        for(let n = 0; n < total; n++) {
            const {
                nabla_w,
                nabla_b
            } = this.getMeanGradient(input[n], target[n]);
            this.weights = this.weights.map((w0, k) => w0.map((w1, i) => w1.map((w, j) => w - nabla_w[k][i][j])));
            this.biases = this.biases.map((b0, k) => b0.map((b, i) => b - nabla_b[k][i]));
        }
    }
}