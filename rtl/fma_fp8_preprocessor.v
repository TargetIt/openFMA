//============================================================================
// FP8 Preprocessor (4-Lane)
// Unpacks 4 FP8 (E4M3) values from 32-bit input
// Performs per-lane multiplication and prepares for CSA tree
//============================================================================
module fma_fp8_preprocessor (
    input  wire [31:0] A,           // 4 packed FP8 values
    input  wire [31:0] B,           // 4 packed FP8 values
    input  wire [31:0] C,           // 4 packed FP8 values

    // Per-lane products (8-bit each)
    output wire [7:0]  product_0,
    output wire [7:0]  product_1,
    output wire [7:0]  product_2,
    output wire [7:0]  product_3,

    // Per-lane exponent sums (extended to FP32 range)
    output wire [7:0]  exp_sum_0,
    output wire [7:0]  exp_sum_1,
    output wire [7:0]  exp_sum_2,
    output wire [7:0]  exp_sum_3,

    // Per-lane signs
    output wire [3:0]  signs_product,
    output wire [3:0]  signs_c,

    // Per-lane C mantissa
    output wire [3:0]  man_c_0,
    output wire [3:0]  man_c_1,
    output wire [3:0]  man_c_2,
    output wire [3:0]  man_c_3,

    // Per-lane C exponent (FP32 range)
    output wire [7:0]  exp_c_0,
    output wire [7:0]  exp_c_1,
    output wire [7:0]  exp_c_2,
    output wire [7:0]  exp_c_3
);

    // Unpack lanes
    wire [7:0] a_lane [0:3];
    wire [7:0] b_lane [0:3];
    wire [7:0] c_lane [0:3];

    assign a_lane[0] = A[7:0];
    assign a_lane[1] = A[15:8];
    assign a_lane[2] = A[23:16];
    assign a_lane[3] = A[31:24];

    assign b_lane[0] = B[7:0];
    assign b_lane[1] = B[15:8];
    assign b_lane[2] = B[23:16];
    assign b_lane[3] = B[31:24];

    assign c_lane[0] = C[7:0];
    assign c_lane[1] = C[15:8];
    assign c_lane[2] = C[23:16];
    assign c_lane[3] = C[31:24];

    // Per-lane field extraction and multiplication
    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : fp8_lane
            wire        sign_a = a_lane[i][7];
            wire [3:0]  exp_a  = a_lane[i][6:3];
            wire [2:0]  man_a  = a_lane[i][2:0];

            wire        sign_b = b_lane[i][7];
            wire [3:0]  exp_b  = b_lane[i][6:3];
            wire [2:0]  man_b  = b_lane[i][2:0];

            wire        sign_c_l = c_lane[i][7];
            wire [3:0]  exp_c_l  = c_lane[i][6:3];
            wire [2:0]  man_c_l  = c_lane[i][2:0];

            // Mantissa with implicit bit
            wire [3:0]  full_man_a = (exp_a == 4'b0) ? {1'b0, man_a} : {1'b1, man_a};
            wire [3:0]  full_man_b = (exp_b == 4'b0) ? {1'b0, man_b} : {1'b1, man_b};
            wire [3:0]  full_man_c = (exp_c_l == 4'b0) ? {1'b0, man_c_l} : {1'b1, man_c_l};

            // 4x4 multiplication -> 8-bit product
            wire [7:0]  prod = full_man_a * full_man_b;

            // Exponent sum with FP32 bias extension (+113 = 120 + 7 - 7 - 7 + 127 - 127)
            wire [7:0]  exp_s = {4'b0, exp_a} + {4'b0, exp_b} - 8'd7 + 8'd113;
            wire [7:0]  exp_c_ext = (exp_c_l == 4'b0) ? 8'd0 : ({4'b0, exp_c_l} + 8'd113);
        end
    endgenerate

    // Map generate outputs
    assign product_0 = fp8_lane[0].prod;
    assign product_1 = fp8_lane[1].prod;
    assign product_2 = fp8_lane[2].prod;
    assign product_3 = fp8_lane[3].prod;

    assign exp_sum_0 = fp8_lane[0].exp_s;
    assign exp_sum_1 = fp8_lane[1].exp_s;
    assign exp_sum_2 = fp8_lane[2].exp_s;
    assign exp_sum_3 = fp8_lane[3].exp_s;

    assign signs_product = {a_lane[3][7] ^ b_lane[3][7],
                            a_lane[2][7] ^ b_lane[2][7],
                            a_lane[1][7] ^ b_lane[1][7],
                            a_lane[0][7] ^ b_lane[0][7]};

    assign signs_c = {c_lane[3][7], c_lane[2][7], c_lane[1][7], c_lane[0][7]};

    assign man_c_0 = fp8_lane[0].full_man_c;
    assign man_c_1 = fp8_lane[1].full_man_c;
    assign man_c_2 = fp8_lane[2].full_man_c;
    assign man_c_3 = fp8_lane[3].full_man_c;

    assign exp_c_0 = fp8_lane[0].exp_c_ext;
    assign exp_c_1 = fp8_lane[1].exp_c_ext;
    assign exp_c_2 = fp8_lane[2].exp_c_ext;
    assign exp_c_3 = fp8_lane[3].exp_c_ext;

endmodule
