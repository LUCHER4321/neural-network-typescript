import { Activation } from "./types";

export const sigmoid: Activation = {
    activation: (x: number) => 1 / (1 + Math.exp(-x)),
    dActivation: (x: number) => {
        const sg = sigmoid.activation(x);
        return sg * (1 - sg);
    }
}

export const reLU: Activation = {
    activation: (x: number) => Math.max(0, x),
    dActivation: (x: number) => x >= 0 ? 1 : 0
}

export const leakyReLU = (alpha: number = 0.01): Activation => {
    if(alpha >= 1 || alpha <= 0) throw new Error("Alpha must be greater than 0 and lower than 1");
    return {
        activation: (x: number) => Math.max(x, alpha * x),
        dActivation: (x: number) => x >= 0 ? 1 : alpha
    }
};

export const linear: Activation = {
    activation: (x: number) => x,
    dActivation: () => 1
}

export const tanh: Activation = {
    activation: (x: number) => 2 * sigmoid.activation(2 * x) - 1,
    dActivation: (x: number) => 4 * sigmoid.dActivation(2 * x)
}

export const softPlus: Activation = {
    activation: (x: number) => Math.log(1 + Math.exp(x)),
    dActivation: (x: number) => sigmoid.activation(x)
}