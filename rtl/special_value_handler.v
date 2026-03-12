//============================================================================
// Special Value Handler
// Handles NaN, Inf, Zero, Denormal for FP32/FP16/FP8
//============================================================================
module special_value_handler (
    input  wire [2:0]  mode,
    input  wire [31:0] A,
    input  wire [31:0] B,
    input  wire [31:0] C,
    output reg         is_special,      // Special case detected
    output reg  [31:0] special_result,  // Result for special case
    output reg         special_overflow
);

    // FP32 field extraction
    wire        a_sign_fp32 = A[31];
    wire [7:0]  a_exp_fp32  = A[30:23];
    wire [22:0] a_man_fp32  = A[22:0];
    wire        b_sign_fp32 = B[31];
    wire [7:0]  b_exp_fp32  = B[30:23];
    wire [22:0] b_man_fp32  = B[22:0];
    wire        c_sign_fp32 = C[31];
    wire [7:0]  c_exp_fp32  = C[30:23];
    wire [22:0] c_man_fp32  = C[22:0];

    // FP32 special value detection
    wire a_is_nan_fp32  = (a_exp_fp32 == 8'hFF) && (a_man_fp32 != 23'b0);
    wire a_is_inf_fp32  = (a_exp_fp32 == 8'hFF) && (a_man_fp32 == 23'b0);
    wire a_is_zero_fp32 = (a_exp_fp32 == 8'h00) && (a_man_fp32 == 23'b0);
    wire b_is_nan_fp32  = (b_exp_fp32 == 8'hFF) && (b_man_fp32 != 23'b0);
    wire b_is_inf_fp32  = (b_exp_fp32 == 8'hFF) && (b_man_fp32 == 23'b0);
    wire b_is_zero_fp32 = (b_exp_fp32 == 8'h00) && (b_man_fp32 == 23'b0);
    wire c_is_nan_fp32  = (c_exp_fp32 == 8'hFF) && (c_man_fp32 != 23'b0);
    wire c_is_inf_fp32  = (c_exp_fp32 == 8'hFF) && (c_man_fp32 == 23'b0);
    wire c_is_zero_fp32 = (c_exp_fp32 == 8'h00) && (c_man_fp32 == 23'b0);

    wire prod_sign_fp32 = a_sign_fp32 ^ b_sign_fp32;

    // FP16 field extraction
    wire        a_sign_fp16 = A[15];
    wire [4:0]  a_exp_fp16  = A[14:10];
    wire [9:0]  a_man_fp16  = A[9:0];
    wire        b_sign_fp16 = B[15];
    wire [4:0]  b_exp_fp16  = B[14:10];
    wire [9:0]  b_man_fp16  = B[9:0];
    wire        c_sign_fp16 = C[15];
    wire [4:0]  c_exp_fp16  = C[14:10];
    wire [9:0]  c_man_fp16  = C[9:0];

    wire a_is_nan_fp16  = (a_exp_fp16 == 5'h1F) && (a_man_fp16 != 10'b0);
    wire a_is_inf_fp16  = (a_exp_fp16 == 5'h1F) && (a_man_fp16 == 10'b0);
    wire a_is_zero_fp16 = (a_exp_fp16 == 5'h00) && (a_man_fp16 == 10'b0);
    wire b_is_nan_fp16  = (b_exp_fp16 == 5'h1F) && (b_man_fp16 != 10'b0);
    wire b_is_inf_fp16  = (b_exp_fp16 == 5'h1F) && (b_man_fp16 == 10'b0);
    wire b_is_zero_fp16 = (b_exp_fp16 == 5'h00) && (b_man_fp16 == 10'b0);
    wire c_is_nan_fp16  = (c_exp_fp16 == 5'h1F) && (c_man_fp16 != 10'b0);
    wire c_is_inf_fp16  = (c_exp_fp16 == 5'h1F) && (c_man_fp16 == 10'b0);
    wire c_is_zero_fp16 = (c_exp_fp16 == 5'h00) && (c_man_fp16 == 10'b0);

    wire prod_sign_fp16 = a_sign_fp16 ^ b_sign_fp16;

    // Canonical NaN constants
    localparam FP32_NAN = 32'h7FC00000;
    localparam FP16_NAN = 16'h7E00;

    always @(*) begin
        is_special       = 1'b0;
        special_result   = 32'b0;
        special_overflow = 1'b0;

        case (mode)
        3'b000: begin // FP32
            if (a_is_nan_fp32 || b_is_nan_fp32 || c_is_nan_fp32) begin
                is_special     = 1'b1;
                special_result = FP32_NAN;
            end else if (a_is_inf_fp32 && b_is_zero_fp32 ||
                         b_is_inf_fp32 && a_is_zero_fp32) begin
                // Inf * 0 = NaN
                is_special     = 1'b1;
                special_result = FP32_NAN;
            end else if ((a_is_inf_fp32 || b_is_inf_fp32) && c_is_inf_fp32 &&
                         (prod_sign_fp32 != c_sign_fp32)) begin
                // Inf + (-Inf) = NaN
                is_special     = 1'b1;
                special_result = FP32_NAN;
            end else if (a_is_inf_fp32 || b_is_inf_fp32) begin
                is_special     = 1'b1;
                special_result = {prod_sign_fp32, 8'hFF, 23'b0};
            end else if (c_is_inf_fp32) begin
                is_special     = 1'b1;
                special_result = C;
            end else if (a_is_zero_fp32 || b_is_zero_fp32) begin
                is_special     = 1'b1;
                special_result = C;
            end
        end

        3'b001, 3'b100: begin // FP16 / FP16->FP32 Acc
            if (a_is_nan_fp16 || b_is_nan_fp16 || c_is_nan_fp16) begin
                is_special     = 1'b1;
                special_result = {16'b0, FP16_NAN};
            end else if (a_is_inf_fp16 && b_is_zero_fp16 ||
                         b_is_inf_fp16 && a_is_zero_fp16) begin
                is_special     = 1'b1;
                special_result = {16'b0, FP16_NAN};
            end else if ((a_is_inf_fp16 || b_is_inf_fp16) && c_is_inf_fp16 &&
                         (prod_sign_fp16 != c_sign_fp16)) begin
                is_special     = 1'b1;
                special_result = {16'b0, FP16_NAN};
            end else if (a_is_inf_fp16 || b_is_inf_fp16) begin
                is_special     = 1'b1;
                special_result = {16'b0, prod_sign_fp16, 5'h1F, 10'b0};
            end else if (c_is_inf_fp16) begin
                is_special     = 1'b1;
                special_result = {16'b0, C[15:0]};
            end else if (a_is_zero_fp16 || b_is_zero_fp16) begin
                is_special     = 1'b1;
                special_result = {16'b0, C[15:0]};
            end
        end

        default: begin
            // FP8, INT8, accumulate modes: no special handling at this level
            is_special     = 1'b0;
            special_result = 32'b0;
        end
        endcase
    end

endmodule
