// =============================================================================
// Testbench for Stage 0: FP32 FMA
// =============================================================================
`timescale 1ns / 1ps

module tb_fma_fp32_stage0;

    reg         clk;
    reg         rst_n;
    reg         valid_in;
    reg  [31:0] A, B, C;
    wire [31:0] result;
    wire        valid_out;
    wire        overflow;
    wire        underflow;
    wire        inexact;

    // Instantiate DUT
    fma_fp32_stage0 dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .A         (A),
        .B         (B),
        .C         (C),
        .result    (result),
        .valid_out (valid_out),
        .overflow  (overflow),
        .underflow (underflow),
        .inexact   (inexact)
    );

    // Clock generation: 10ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // Helper: real to FP32 bits (using system functions)
    // FP32 constants for testing
    localparam FP32_ONE       = 32'h3F800000; // 1.0
    localparam FP32_TWO       = 32'h40000000; // 2.0
    localparam FP32_THREE     = 32'h40400000; // 3.0
    localparam FP32_FOUR      = 32'h40800000; // 4.0
    localparam FP32_FIVE      = 32'h40A00000; // 5.0
    localparam FP32_ZERO      = 32'h00000000; // 0.0
    localparam FP32_NEG_ONE   = 32'hBF800000; // -1.0
    localparam FP32_HALF      = 32'h3F000000; // 0.5
    localparam FP32_TEN       = 32'h41200000; // 10.0
    localparam FP32_INF       = 32'h7F800000; // +Inf
    localparam FP32_NEG_INF   = 32'hFF800000; // -Inf
    localparam FP32_NAN       = 32'h7FC00000; // NaN (quiet)

    integer test_count;
    integer pass_count;
    integer fail_count;

    task apply_input;
        input [31:0] a_val, b_val, c_val;
        begin
            @(posedge clk);
            A <= a_val;
            B <= b_val;
            C <= c_val;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
        end
    endtask

    task wait_result;
        begin
            // Wait for 3-stage pipeline (3 cycles after input)
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
            // NaN: exp = 0xFF, mantissa != 0
            if (result[30:23] == 8'hFF && result[22:0] != 23'h0) begin
                $display("PASS: %0s - result = %h (NaN)", test_name, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected NaN)", test_name, result);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_inf;
        input expected_sign;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result == {expected_sign, 8'hFF, 23'h0}) begin
                $display("PASS: %0s - result = %h (Inf)", test_name, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected %cInf)", test_name, result, expected_sign ? "-" : "+");
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("stage0_test.vcd");
        $dumpvars(0, tb_fma_fp32_stage0);

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        rst_n = 0;
        valid_in = 0;
        A = 0; B = 0; C = 0;

        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("=== Stage 0: FP32 FMA Testbench ===");

        // Test 1: 1.0 * 1.0 + 0.0 = 1.0
        apply_input(FP32_ONE, FP32_ONE, FP32_ZERO);
        wait_result();
        check_result(FP32_ONE, "1.0 * 1.0 + 0.0");

        // Test 2: 2.0 * 3.0 + 4.0 = 10.0
        apply_input(FP32_TWO, FP32_THREE, FP32_FOUR);
        wait_result();
        check_result(FP32_TEN, "2.0 * 3.0 + 4.0");

        // Test 3: 1.0 * 1.0 + 1.0 = 2.0
        apply_input(FP32_ONE, FP32_ONE, FP32_ONE);
        wait_result();
        check_result(FP32_TWO, "1.0 * 1.0 + 1.0");

        // Test 4: NaN input -> NaN output
        apply_input(FP32_NAN, FP32_ONE, FP32_ZERO);
        wait_result();
        check_nan("NaN * 1.0 + 0.0");

        // Test 5: 0 * Inf = NaN
        apply_input(FP32_ZERO, FP32_INF, FP32_ZERO);
        wait_result();
        check_nan("0.0 * Inf + 0.0");

        // Test 6: Inf + (-Inf) = NaN
        apply_input(FP32_ONE, FP32_INF, FP32_NEG_INF);
        wait_result();
        check_nan("1.0 * Inf + (-Inf)");

        // Test 7: 1.0 * Inf + 0.0 = +Inf
        apply_input(FP32_ONE, FP32_INF, FP32_ZERO);
        wait_result();
        check_inf(1'b0, "1.0 * Inf + 0.0");

        // Test 8: 0.0 * 0.0 + 0.0 = 0.0
        apply_input(FP32_ZERO, FP32_ZERO, FP32_ZERO);
        wait_result();
        check_result(FP32_ZERO, "0.0 * 0.0 + 0.0");

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
