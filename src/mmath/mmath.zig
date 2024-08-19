const std = @import("std");
const math = std.math;
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;

const MatrixError = error{
    DimensionMismatch,
    AllocationFailure,
    InvalidDimensions,
};

pub fn Matrix(comptime T: type) type {
    return struct {
        data: [][]T,
        rows: usize,
        cols: usize,

        const Self = @This();

        pub fn init(comptime data: anytype, allocator: Allocator) !Self {
            const info = @typeInfo(@TypeOf(data));
            if (info != .Array or @typeInfo(info.Array.child) != .Array) {
                @compileError("Expected a 2D array");
            }
            const rows = data.len;
            const cols = data[0].len;
            var new_data = try allocator.alloc([]T, rows);
            errdefer allocator.free(new_data);
            for (new_data, 0..) |*row, i| {
                row.* = try allocator.alloc(T, cols);
                errdefer {
                    for (new_data[0..i]) |r| allocator.free(r);
                    allocator.free(new_data);
                }
                @memcpy(row.*, &data[i]);
            }
            return Self{
                .data = new_data,
                .rows = rows,
                .cols = cols,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            for (self.data) |row| {
                allocator.free(row);
            }
            allocator.free(self.data);
            self.* = undefined;
        }

        pub fn init_empty(rows: usize, cols: usize, allocator: Allocator) !@This() {
            const data = try allocator.alloc([]T, rows);
            errdefer allocator.free(data);

            for (data) |*row| {
                row.* = try allocator.alloc(T, cols);
            }

            return @This(){ .data = data, .rows = rows, .cols = cols };
        }
    };
}

pub fn get_rand_generator() !std.Random {
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

pub fn multiply_matrix(comptime T: type, m1: *const Matrix(T), m2: *const Matrix(T), allocator: Allocator) !Matrix(T) {
    if (m1.cols != m2.rows) {
        return error.DimensionMismatch;
    }

    var result = try Matrix(T).init_empty(m1.rows, m2.cols, allocator);
    errdefer result.deinit();

    for (0..m1.rows) |i| {
        for (0..m2.cols) |j| {
            result.data[i][j] = 0;
            for (0..m1.cols) |k| {
                result.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }

    return result;
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

test "matrix" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [2][3]f64{
        [_]f64{ 1.123, 2.4123, 3.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
    };

    var my_matrix = try Matrix(f64).init(data, allocator);
    defer my_matrix.deinit(allocator);

    for (my_matrix.data) |rows| {
        for (rows) |col| {
            std.debug.print("{}\n", .{col});
        }
    }
}

test "matrix_multiply" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [3][3]f64{
        [_]f64{ 1.123, 2.4123, 3.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
        [_]f64{ 4.12354, 5.41243, 6.12345 },
    };

    var my_matrix = try Matrix(f64).init(data, allocator);
    defer my_matrix.deinit(allocator);

    var m2 = try Matrix(f64).init(data, allocator);
    defer m2.deinit(allocator);

    var v = try multiply_matrix(f64, &my_matrix, &m2, allocator);
    defer v.deinit(allocator);

    for (v.data) |rows| {
        std.debug.print("ROW\n", .{});
        for (rows) |col| {
            std.debug.print("{}\n", .{col});
        }
    }
}
