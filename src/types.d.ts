export interface Gradient {
    nabla_w: number[][][],
    nabla_b: number[][]
}

export type NumericFunction = (x: number) => number;

export interface Activation {
    activation: NumericFunction,
    dActivation: NumericFunction
}