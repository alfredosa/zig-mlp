const std = @import("std");

const MatrixError = error{
    DimensionMismatch,
    AllocationFailure,
    InvalidDimensions,
};

fn multiplyMatrix(comptime X: usize, comptime Y: usize, m1: *const [X][Y]f64, comptime X2: usize, comptime Y2: usize, m2: *const [X2][Y2]f64) ![X][Y2]f64 {
    if (Y != X2) {
        return MatrixError.DimensionMismatch;
    }
    var result: [X][Y2]f64 = undefined;
    for (0..X) |i| {
        for (0..Y2) |j| {
            var sum: f64 = 0;
            for (0..Y) |k| {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

test "matrix multiplication" {
    var data = [3][3]f64{
        [_]f64{ 1.123, 2.4123, 3.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
    };
    const result = try multiplyMatrix(3, 3, &data, 3, 3, &data);

    std.debug.print("Result:\n", .{});
    for (result) |row| {
        for (row) |val| {
            std.debug.print("{d:.9} ", .{val});
        }
        std.debug.print("\n", .{});
    }

    const expected = [3][3]f64{
        [_]f64{ 24.087558389175, 32.670350348185, 37.404968806325 },
        [_]f64{ 52.198841294725, 72.383728231835, 83.518361301675 },
        [_]f64{ 52.198841294725, 72.383728231835, 83.518361301675 },
    };

    std.debug.print("\nChecking results:\n", .{});
    for (result, 0..) |row, i| {
        for (row, 0..) |val, j| {
            const diff = @abs(val - expected[i][j]);
            std.debug.print("Difference at [{},{}]: {d:.9}\n", .{ i, j, diff });
            try std.testing.expectApproxEqAbs(val, expected[i][j], 1e-9);
        }
    }
}
