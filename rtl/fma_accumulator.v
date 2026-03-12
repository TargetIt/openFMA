//============================================================================
// Mixed-Precision Accumulator
// Supports:
//   - FP16 x FP16 -> FP32 accumulation (mode=100)
//   - FP8x4 -> FP32 accumulation (mode=101)
//============================================================================
module fma_accumulator (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        acc_en,          // Accumulate enable
    input  wire [31:0] acc_in,          // External accumulator initial value
    input  wire [31:0] fma_result,      // FMA computation result
    input  wire        valid_in,        // Valid signal from pipeline
    input  wire [2:0]  mode,

    output reg  [31:0] acc_out,         // Accumulator output
    output reg         acc_overflow     // Accumulator overflow
);

    //------------------------------------------------------------------------
    // Accumulator Register
    //------------------------------------------------------------------------
    reg [31:0] acc_reg;

    // Detect FP32 overflow (Inf)
    wire is_inf = (fma_result[30:23] == 8'hFF);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg      <= 32'b0;
            acc_out      <= 32'b0;
            acc_overflow <= 1'b0;
        end else begin
            if (acc_en && valid_in && (mode == 3'b100 || mode == 3'b101)) begin
                if (is_inf) begin
                    // Saturation: keep at positive infinity
                    acc_reg      <= 32'h7F800000;
                    acc_overflow <= 1'b1;
                end else begin
                    acc_reg      <= fma_result;
                    acc_overflow <= 1'b0;
                end
            end else if (!acc_en) begin
                // Load initial value when accumulation is not active
                acc_reg      <= acc_in;
                acc_overflow <= 1'b0;
            end
            acc_out <= acc_reg;
        end
    end

endmodule
