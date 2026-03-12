//============================================================================
// 48-bit Shared Multiplier
// Supports FP32 (24x24), FP16 (11x11), FP8 (4x4), INT8 (8x8) modes
//============================================================================
module multiplier_48bit (
    input  wire [23:0] mul_a,       // Multiplicand (with implicit 1)
    input  wire [23:0] mul_b,       // Multiplier   (with implicit 1)
    output wire [47:0] product      // 48-bit product
);

    assign product = mul_a * mul_b;

endmodule
