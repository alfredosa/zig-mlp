const std = @import("std");
const math = std.math;

const mymath = @import("../mmath/mmath.zig");

const Allocator = std.mem.Allocator;

// TODO: Verify the beauty of weights
pub const Neuron = struct {
    weights: []f64,
    bias: f64,
    num_inputs: usize,

    pub fn init(allocator: Allocator, num_inputs: usize, random: *std.rand.Random) !Neuron {
        const weights = try allocator.alloc(f64, num_inputs);
        for (weights) |*weight| {
            weight.* = random_bzao(random);
        }
        return Neuron{
            .weights = weights,
            .bias = random_bzao(random),
            .num_inputs = num_inputs,
        };
    }

    pub fn deinit(self: *Neuron, allocator: Allocator) void {
        allocator.free(self.weights);
    }
};

// Random value between zero and one :)
pub fn random_bzao(random: *std.rand.Random) f64 {
    return random.float(f64) * 2 - 1;
}

pub const Layer = struct {
    neurons: []Neuron,
    inputs: []f64,
    outputs: []f64,

    pub fn init(allocator: Allocator, num_neurons: usize, num_inputs_per_neuron: usize, random: *std.rand.Random) !Layer {
        const neurons = try allocator.alloc(Neuron, num_neurons);
        for (neurons) |*neuron| {
            neuron.* = try Neuron.init(allocator, num_inputs_per_neuron, random);
        }
        return Layer{
            .neurons = neurons,
            .inputs = try allocator.alloc(f64, num_inputs_per_neuron),
            .outputs = try allocator.alloc(f64, num_neurons),
        };
    }

    pub fn deinit(self: *Layer, allocator: Allocator) void {
        for (self.neurons) |*neuron| {
            neuron.deinit(allocator);
        }
        allocator.free(self.neurons);
        allocator.free(self.inputs);
        allocator.free(self.outputs);
    }
};

pub const NeuralNetwork = struct {
    inputs: []f64,
    layers: []Layer,

    pub fn init(allocator: Allocator, layer_sizes: []const usize) !NeuralNetwork {
        var random = mymath.get_rand_generator();
        const layers = try allocator.alloc(Layer, layer_sizes.len - 1);

        for (layers, 0..) |*layer, i| {
            layer.* = try Layer.init(allocator, layer_sizes[i + 1], layer_sizes[i], &random);
        }

        return NeuralNetwork{
            .inputs = try allocator.alloc(f64, layer_sizes[0]),
            .layers = layers,
        };
    }

    pub fn deinit(self: *NeuralNetwork, allocator: Allocator) void {
        for (self.layers) |*layer| {
            layer.deinit(allocator);
        }
        allocator.free(self.layers);
        allocator.free(self.inputs);
    }
};
