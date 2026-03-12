//============================================================================
// INT8 Datapath
// Implements signed INT8 multiply-add: result = A_int * B_int + C_int
// Reuses FP16 lane multiplier with sign-extension control
//============================================================================
module fma_int8_datapath (
    input  wire [7:0]  A_int,       // Signed INT8 multiplicand
    input  wire [7:0]  B_int,       // Signed INT8 multiplier
    input  wire [15:0] C_int,       // Signed INT16 addend

    output wire [15:0] result,      // Signed INT16 result
    output wire        overflow     // Overflow flag
);

    //------------------------------------------------------------------------
    // Signed multiplication: A_int * B_int -> 16-bit signed product
    //------------------------------------------------------------------------
    wire signed [7:0]  a_signed = A_int;
    wire signed [7:0]  b_signed = B_int;
    wire signed [15:0] product  = a_signed * b_signed;

    //------------------------------------------------------------------------
    // Signed addition: product + C_int -> 16-bit result with overflow check
    //------------------------------------------------------------------------
    wire signed [15:0] c_signed = C_int;
    wire signed [16:0] sum_ext  = {product[15], product} + {c_signed[15], c_signed};

    assign result = sum_ext[15:0];

    // Overflow: sign of extended sum doesn't match truncated result
    assign overflow = (sum_ext[16] != sum_ext[15]);

endmodule
