//============================================================================
// 3-2 Carry Save Adder (50-bit)
// Compresses 3 inputs into 2 outputs (sum + carry)
// Extensible to 4-2 CSA for FP8 mode
//============================================================================
module csa_3_2 #(
    parameter WIDTH = 50
)(
    input  wire [WIDTH-1:0] in_a,   // Input A
    input  wire [WIDTH-1:0] in_b,   // Input B
    input  wire [WIDTH-1:0] in_c,   // Input C
    output wire [WIDTH-1:0] sum,    // Sum output
    output wire [WIDTH-1:0] carry   // Carry output (shifted left by 1)
);

    assign sum   = in_a ^ in_b ^ in_c;
    assign carry = {(in_a & in_b | in_b & in_c | in_a & in_c), 1'b0};

endmodule
