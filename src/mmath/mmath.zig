const std = @import("std");
const math = std.math;
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;

pub fn get_rand_generator() std.Random {
    const rand = std.crypto.random;
    return rand;
}

pub fn sigmoid(x: anytype) @TypeOf(x) {
    return 1.0 / (1.0 + math.exp(-x));
}

pub fn relu(x: anytype) @TypeOf(x) {
    return if (x > 0.0) x else 0.0;
}

pub fn tahn(x: anytype) @TypeOf(x) {
    return ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)));
}

test "test_sigmoid" {
    const x = 1.123;
    const r = sigmoid(x);
    try expect(r > 0.74 and r < 0.77);
}

test "relu" {
    const x = 0.123;
    const x2 = -0.021;
    try expect(relu(x) == x and relu(x2) == 0.0);
}

test "tahn" {
    const x = 0.123;
    try expect(tahn(x) > 0.122 and tahn(x) < 0.123);
}
