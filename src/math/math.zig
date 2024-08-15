const std = @import("std");

pub fn get_rand_generator() !std.Random {
    const rand = std.crypto.random;
    return rand;
}
