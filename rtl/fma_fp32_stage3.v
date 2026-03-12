//============================================================================
// FMA Pipeline Stage 3: Normalization, Rounding, and Result Assembly
// - Adds CSA sum and carry
// - Normalizes via LZD
// - Rounds (round to nearest even)
// - Assembles final result
//============================================================================
module fma_fp32_stage3 (
    input  wire        clk,
    input  wire        rst_n,

    // From Stage 2
    input  wire        valid_s2,
    input  wire [2:0]  mode_s2,
    input  wire        sign_result,
    input  wire [9:0]  exp_tentative,
    input  wire [49:0] csa_sum,
    input  wire [49:0] csa_carry,
    input  wire        eff_sub,
    input  wire        sticky_s2,
    input  wire [31:0] C_passthrough_s2,

    // Outputs (registered)
    output reg         valid_out,
    output reg  [31:0] result,
    output reg         overflow,
    output reg         underflow,
    output reg         inexact
);

    //------------------------------------------------------------------------
    // 50-bit Adder: sum + carry
    //------------------------------------------------------------------------
    wire [50:0] add_result;

    adder_50bit u_adder (
        .in_a   (csa_sum),
        .in_b   (csa_carry),
        .cin    (1'b0),
        .result (add_result)
    );

    //------------------------------------------------------------------------
    // Handle sign correction for effective subtraction
    //------------------------------------------------------------------------
    wire [50:0] abs_result;
    wire        final_sign;

    // If result is negative (MSB set after effective subtraction), negate
    assign abs_result = (eff_sub && add_result[50]) ?
                        (~add_result + 51'b1) : add_result;
    assign final_sign = (eff_sub && add_result[50]) ?
                        ~sign_result : sign_result;

    //------------------------------------------------------------------------
    // Normalizer / LZD
    //------------------------------------------------------------------------
    wire [49:0] normalized;
    wire [7:0]  shift_count;
    wire        is_zero;

    normalizer_lzd u_norm (
        .data_in     (abs_result),
        .normalized  (normalized),
        .shift_count (shift_count),
        .is_zero     (is_zero)
    );

    //------------------------------------------------------------------------
    // Exponent Adjustment
    // Binary point is between bits 48 and 47 in the internal format.
    // Normalizer shifts MSB to bit 49, so exp = exp_tentative + 1 - shift_count.
    // For overflow case (right shift), exp = exp_tentative + 2.
    //------------------------------------------------------------------------
    wire [9:0] exp_adjusted;

    assign exp_adjusted = (shift_count == 8'd255) ?
                          (exp_tentative + 10'd2) :            // Overflow: right shift by 1
                          (exp_tentative + 10'd1 - {2'b0, shift_count}); // Normal: +1 then subtract LZD

    //------------------------------------------------------------------------
    // Rounding (Round to Nearest Even)
    //------------------------------------------------------------------------
    wire [22:0] mantissa_out;
    wire        guard, round_bit, sticky;
    wire        round_up;

    // After normalization, bit [49] is implicit 1 (not stored).
    // Mantissa bits: [48:26] = 23 stored mantissa bits
    // Guard bit = [25], Round bit = [24], Sticky = OR of [23:0] and sticky_s2
    assign mantissa_out = normalized[48:26];
    assign guard        = normalized[25];
    assign round_bit    = normalized[24];
    assign sticky       = (|normalized[23:0]) | sticky_s2;

    // Round to nearest even
    assign round_up = guard & (round_bit | sticky | mantissa_out[0]);

    wire [23:0] mantissa_rounded = {1'b0, mantissa_out} + {23'b0, round_up};

    // If rounding causes overflow of mantissa, increment exponent
    wire [9:0]  exp_final;
    wire [22:0] man_final;

    assign exp_final = mantissa_rounded[23] ?
                       (exp_adjusted + 10'd1) : exp_adjusted;
    assign man_final = mantissa_rounded[23] ?
                       mantissa_rounded[23:1] : mantissa_rounded[22:0];

    //------------------------------------------------------------------------
    // FP16 result assembly
    //------------------------------------------------------------------------
    wire [15:0] fp16_result;
    wire [9:0]  fp16_exp_adj;
    wire [4:0]  fp16_exp_out;
    wire [9:0]  fp16_man_out;

    assign fp16_exp_adj = exp_final - 10'd112; // Convert back from FP32 range
    assign fp16_exp_out = (fp16_exp_adj[9] || fp16_exp_adj == 10'd0) ? 5'd0 :
                          (fp16_exp_adj >= 10'd31) ? 5'd31 : fp16_exp_adj[4:0];
    assign fp16_man_out = man_final[22:13]; // Top 10 bits of FP32 mantissa
    assign fp16_result  = (fp16_exp_out == 5'd31) ?
                          {final_sign, 5'h1F, 10'b0} :     // Inf
                          {final_sign, fp16_exp_out, fp16_man_out};

    //------------------------------------------------------------------------
    // INT8 result assembly
    //------------------------------------------------------------------------
    wire [15:0] int8_product;
    wire [15:0] int8_result;
    wire        int8_overflow;

    // For INT8 mode, product is in add_result[15:0], C is in C_passthrough_s2[15:0]
    assign int8_product = add_result[15:0];
    assign int8_result  = int8_product + C_passthrough_s2[15:0];

    // Overflow detection: check if result exceeds signed 16-bit range
    // Since both operands are 16-bit signed, overflow occurs when signs match
    // but result sign differs
    assign int8_overflow = (int8_product[15] == C_passthrough_s2[15]) &&
                           (int8_result[15] != int8_product[15]);

    //------------------------------------------------------------------------
    // FP8 result assembly (Lane 0 only; other lanes in fp8_preprocessor)
    //------------------------------------------------------------------------
    wire [7:0]  fp8_result;
    wire [9:0]  fp8_exp_adj;
    wire [3:0]  fp8_exp_out;
    wire [2:0]  fp8_man_out;

    assign fp8_exp_adj = exp_final - 10'd113;
    assign fp8_exp_out = (fp8_exp_adj[9] || fp8_exp_adj == 10'd0) ? 4'd0 :
                         (fp8_exp_adj >= 10'd15) ? 4'd15 : fp8_exp_adj[3:0];
    assign fp8_man_out = man_final[22:20];
    assign fp8_result  = (fp8_exp_out == 4'd15) ?
                         {final_sign, 4'hF, 3'b0} :
                         {final_sign, fp8_exp_out, fp8_man_out};

    //------------------------------------------------------------------------
    // Overflow / Underflow / Inexact Detection (FP32)
    //------------------------------------------------------------------------
    wire fp32_overflow  = (exp_final >= 10'd255) && !is_zero;
    wire fp32_underflow = (exp_final[9]) || (exp_final == 10'd0 && !is_zero);
    wire fp32_inexact   = guard | round_bit | sticky;

    //------------------------------------------------------------------------
    // Output Mux & Pipeline Register
    //------------------------------------------------------------------------
    reg [31:0] result_mux;
    reg        ovf_mux, udf_mux, inx_mux;

    always @(*) begin
        result_mux = 32'd0;
        ovf_mux    = 1'b0;
        udf_mux    = 1'b0;
        inx_mux    = 1'b0;

        case (mode_s2)
        3'b000: begin // FP32
            if (is_zero) begin
                result_mux = {final_sign, 31'b0};
            end else if (fp32_overflow) begin
                result_mux = {final_sign, 8'hFF, 23'b0}; // Inf
                ovf_mux    = 1'b1;
            end else if (fp32_underflow) begin
                result_mux = {final_sign, 31'b0}; // Flush to zero
                udf_mux    = 1'b1;
            end else begin
                result_mux = {final_sign, exp_final[7:0], man_final};
            end
            inx_mux = fp32_inexact;
        end

        3'b001: begin // FP16
            if (is_zero) begin
                result_mux = {16'b0, final_sign, 15'b0};
            end else begin
                result_mux = {16'b0, fp16_result};
            end
            ovf_mux = (fp16_exp_out == 5'd31);
            inx_mux = fp32_inexact;
        end

        3'b010: begin // FP8x4 (Lane 0 only here)
            if (is_zero) begin
                result_mux = {24'b0, final_sign, 7'b0};
            end else begin
                result_mux = {24'b0, fp8_result};
            end
            ovf_mux = (fp8_exp_out == 4'd15);
            inx_mux = fp32_inexact;
        end

        3'b011: begin // INT8
            result_mux = {16'b0, int8_result};
            ovf_mux    = int8_overflow;
        end

        3'b100: begin // FP16->FP32 Acc
            if (is_zero) begin
                result_mux = {final_sign, 31'b0};
            end else if (fp32_overflow) begin
                result_mux = {final_sign, 8'hFF, 23'b0};
                ovf_mux    = 1'b1;
            end else begin
                result_mux = {final_sign, exp_final[7:0], man_final};
            end
            inx_mux = fp32_inexact;
        end

        3'b101: begin // FP8x4->FP32 Acc
            if (is_zero) begin
                result_mux = {final_sign, 31'b0};
            end else if (fp32_overflow) begin
                result_mux = {final_sign, 8'hFF, 23'b0};
                ovf_mux    = 1'b1;
            end else begin
                result_mux = {final_sign, exp_final[7:0], man_final};
            end
            inx_mux = fp32_inexact;
        end

        default: begin
            result_mux = 32'd0;
        end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            result    <= 32'd0;
            overflow  <= 1'b0;
            underflow <= 1'b0;
            inexact   <= 1'b0;
        end else begin
            valid_out <= valid_s2;
            result    <= result_mux;
            overflow  <= ovf_mux;
            underflow <= udf_mux;
            inexact   <= inx_mux;
        end
    end

endmodule
