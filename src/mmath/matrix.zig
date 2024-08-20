const std = @import("std");

fn multiplyMatrix(m1: []const []const f64, m2: []const []const f64) ![3][3]f64 {
    if (m1.len != 3 or m1[0].len != 3 or m2.len != 3 or m2[0].len != 3) {
        return error.InvalidDimensions;
    }

    var result: [3][3]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            var sum: f64 = 0;
            for (0..3) |k| {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

pub fn main() !void {
    const data = [3][3]f64{
        [_]f64{ 1.123, 2.4123, 3.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
    };

    const result = try multiplyMatrix(&data, &data);

    std.debug.print("Result:\n", .{});
    for (result) |row| {
        for (row) |val| {
            std.debug.print("{d:.9} ", .{val});
        }
        std.debug.print("\n", .{});
    }

    // Expected results
    const expected = [3][3]f64{
        [_]f64{ 24.087558389175, 32.670350348185, 37.404968806325 },
        [_]f64{ 52.198841294725, 72.383728231835, 83.518361301675 },
        [_]f64{ 52.198841294725, 72.383728231835, 83.518361301675 },
    };

    std.debug.print("\nChecking results:\n", .{});
    for (result, 0..) |row, i| {
        for (row, 0..) |val, j| {
            const diff = @abs(val - expected[i][j]);
            std.debug.print("Difference at [{},{}]: {d:.9e}\n", .{ i, j, diff });
        }
    }
}
