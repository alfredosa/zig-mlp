const std = @import("std");
const nn = @import("nn/nn.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var network = try nn.NeuralNetwork.init(allocator, &[_]usize{ 2, 3, 1 });
    defer network.deinit(allocator);

    // Use your neural network here...
    for (network.layers) |*layer| {
        std.debug.print("Layer has {} inputs and {} outputs\n", .{ layer.inputs.len, layer.outputs.len });
    }
}
