// =============================================================================
// Testbench for Stage 1: FP32 + FP16 FMA
// =============================================================================
`timescale 1ns / 1ps

module tb_fma_fp16_stage1;

    reg         clk;
    reg         rst_n;
    reg         valid_in;
    reg  [1:0]  mode;
    reg  [31:0] A, B, C;
    wire [31:0] result;
    wire        valid_out;
    wire        overflow;
    wire        underflow;
    wire        inexact;

    fma_fp16_stage1 dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .mode(mode), .A(A), .B(B), .C(C),
        .result(result), .valid_out(valid_out),
        .overflow(overflow), .underflow(underflow), .inexact(inexact)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // FP32 constants
    localparam FP32_ONE   = 32'h3F800000; // 1.0
    localparam FP32_TWO   = 32'h40000000; // 2.0
    localparam FP32_THREE = 32'h40400000; // 3.0
    localparam FP32_FOUR  = 32'h40800000; // 4.0
    localparam FP32_TEN   = 32'h41200000; // 10.0
    localparam FP32_ZERO  = 32'h00000000;
    localparam FP32_NAN   = 32'h7FC00000;
    localparam FP32_INF   = 32'h7F800000;
    localparam FP32_NINF  = 32'hFF800000;

    // FP16 constants (packed in lower 16 bits of 32-bit input)
    localparam FP16_ONE     = 32'h00003C00; // 1.0
    localparam FP16_TWO     = 32'h00004000; // 2.0
    localparam FP16_THREE   = 32'h00004200; // 3.0
    localparam FP16_FOUR    = 32'h00004400; // 4.0
    localparam FP16_TEN     = 32'h00004900; // 10.0
    localparam FP16_ZERO    = 32'h00000000; // 0.0
    localparam FP16_HALF    = 32'h00003800; // 0.5
    localparam FP16_NEG_ONE = 32'h0000BC00; // -1.0
    localparam FP16_NAN     = 32'h00007E00; // NaN
    localparam FP16_INF     = 32'h00007C00; // +Inf
    localparam FP16_NINF    = 32'h0000FC00; // -Inf

    integer test_count = 0;
    integer pass_count = 0;
    integer fail_count = 0;

    task apply_input;
        input [1:0] m;
        input [31:0] a_val, b_val, c_val;
        begin
            @(posedge clk);
            mode <= m;
            A <= a_val; B <= b_val; C <= c_val;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task wait_result;
        begin
            @(posedge clk);
            @(posedge clk);
            @(posedge clk);
        end
    endtask

    task check_result;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result == expected) begin
                $display("PASS: %0s - result = %h (expected %h)", test_name, result, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected %h)", test_name, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_nan;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result[30:23] == 8'hFF && result[22:0] != 23'h0) begin
                $display("PASS: %0s - result = %h (NaN)", test_name, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected NaN)", test_name, result);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_fp16_nan;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result[14:10] == 5'h1F && result[9:0] != 10'h0) begin
                $display("PASS: %0s - result = %h (FP16 NaN)", test_name, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected FP16 NaN)", test_name, result);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("stage1_test.vcd");
        $dumpvars(0, tb_fma_fp16_stage1);

        rst_n = 0; valid_in = 0; mode = 0;
        A = 0; B = 0; C = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("=== Stage 1: FP32 + FP16 FMA Testbench ===");

        // --- FP32 Tests (mode=00, same as Stage 0) ---
        $display("--- FP32 Mode Tests ---");

        apply_input(2'b00, FP32_ONE, FP32_ONE, FP32_ZERO);
        wait_result();
        check_result(FP32_ONE, "FP32: 1.0*1.0+0.0=1.0");

        apply_input(2'b00, FP32_TWO, FP32_THREE, FP32_FOUR);
        wait_result();
        check_result(FP32_TEN, "FP32: 2.0*3.0+4.0=10.0");

        apply_input(2'b00, FP32_NAN, FP32_ONE, FP32_ZERO);
        wait_result();
        check_nan("FP32: NaN*1.0+0.0");

        apply_input(2'b00, FP32_ZERO, FP32_INF, FP32_ZERO);
        wait_result();
        check_nan("FP32: 0*Inf+0");

        // --- FP16 Tests (mode=01) ---
        $display("--- FP16 Mode Tests ---");

        // FP16: 1.0 * 1.0 + 0.0 = 1.0
        apply_input(2'b01, FP16_ONE, FP16_ONE, FP16_ZERO);
        wait_result();
        check_result(FP16_ONE, "FP16: 1.0*1.0+0.0=1.0");

        // FP16: 2.0 * 3.0 + 4.0 = 10.0
        apply_input(2'b01, FP16_TWO, FP16_THREE, FP16_FOUR);
        wait_result();
        check_result(FP16_TEN, "FP16: 2.0*3.0+4.0=10.0");

        // FP16: 1.0 * 1.0 + 1.0 = 2.0
        apply_input(2'b01, FP16_ONE, FP16_ONE, FP16_ONE);
        wait_result();
        check_result(FP16_TWO, "FP16: 1.0*1.0+1.0=2.0");

        // FP16: 0.0 * 0.0 + 0.0 = 0.0
        apply_input(2'b01, FP16_ZERO, FP16_ZERO, FP16_ZERO);
        wait_result();
        check_result(FP16_ZERO, "FP16: 0.0*0.0+0.0=0.0");

        // FP16: NaN
        apply_input(2'b01, FP16_NAN, FP16_ONE, FP16_ZERO);
        wait_result();
        check_fp16_nan("FP16: NaN*1.0+0.0");

        // FP16: 0 * Inf = NaN
        apply_input(2'b01, FP16_ZERO, FP16_INF, FP16_ZERO);
        wait_result();
        check_fp16_nan("FP16: 0*Inf+0");

        // FP16: 1.0 * (-1.0) + 0.0 = -1.0
        apply_input(2'b01, FP16_ONE, FP16_NEG_ONE, FP16_ZERO);
        wait_result();
        check_result(FP16_NEG_ONE, "FP16: 1.0*(-1.0)+0.0=-1.0");

        // Summary
        repeat(5) @(posedge clk);
        $display("=== Test Summary: %0d/%0d passed, %0d failed ===",
                 pass_count, test_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
