// =============================================================================
// Testbench for Stage 3: FP32 + FP16 + FP8x4 FMA + Accumulation
// =============================================================================
`timescale 1ns / 1ps

module tb_fma_acc_stage3;

    reg         clk;
    reg         rst_n;
    reg         valid_in;
    reg  [2:0]  mode;
    reg  [31:0] A, B, C;
    reg         acc_en;
    reg  [31:0] acc_in;
    wire [31:0] result;
    wire        valid_out, overflow, underflow, inexact;
    wire [31:0] acc_out;

    fma_acc_stage3 dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .mode(mode), .A(A), .B(B), .C(C),
        .acc_en(acc_en), .acc_in(acc_in),
        .result(result), .valid_out(valid_out),
        .overflow(overflow), .underflow(underflow), .inexact(inexact),
        .acc_out(acc_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // FP32 constants
    localparam FP32_ZERO  = 32'h00000000;
    localparam FP32_ONE   = 32'h3F800000;
    localparam FP32_TWO   = 32'h40000000;
    localparam FP32_THREE = 32'h40400000;
    localparam FP32_FOUR  = 32'h40800000;
    localparam FP32_TEN   = 32'h41200000;

    // FP16 constants
    localparam FP16_ONE   = 32'h00003C00;
    localparam FP16_TWO   = 32'h00004000;
    localparam FP16_THREE = 32'h00004200;
    localparam FP16_FOUR  = 32'h00004400;
    localparam FP16_TEN   = 32'h00004900;
    localparam FP16_ZERO  = 32'h00000000;

    integer test_count = 0, pass_count = 0, fail_count = 0;

    task apply_input;
        input [2:0] m;
        input [31:0] a_val, b_val, c_val;
        input ae;
        input [31:0] ai;
        begin
            @(posedge clk);
            mode <= m; A <= a_val; B <= b_val; C <= c_val;
            acc_en <= ae; acc_in <= ai;
            valid_in <= 1'b1;
            @(posedge clk);
            valid_in <= 1'b0;
            acc_en <= 0;
        end
    endtask

    task wait_result;
        begin
            @(posedge clk); @(posedge clk); @(posedge clk);
        end
    endtask

    task check_result;
        input [31:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result == expected) begin
                $display("PASS: %0s - result = %h", test_name, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result = %h (expected %h)", test_name, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("stage3_test.vcd");
        $dumpvars(0, tb_fma_acc_stage3);

        rst_n = 0; valid_in = 0; mode = 0;
        A = 0; B = 0; C = 0; acc_en = 0; acc_in = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("=== Stage 3: FMA + Accumulation Testbench ===");

        // --- FP32 regression ---
        $display("--- FP32 Mode Regression ---");
        apply_input(3'b000, FP32_ONE, FP32_ONE, FP32_ZERO, 0, 0);
        wait_result();
        check_result(FP32_ONE, "FP32: 1*1+0=1");

        apply_input(3'b000, FP32_TWO, FP32_THREE, FP32_FOUR, 0, 0);
        wait_result();
        check_result(FP32_TEN, "FP32: 2*3+4=10");

        // --- FP16 regression ---
        $display("--- FP16 Mode Regression ---");
        apply_input(3'b001, FP16_ONE, FP16_ONE, FP16_ZERO, 0, 0);
        wait_result();
        check_result(FP16_ONE, "FP16: 1*1+0=1");

        apply_input(3'b001, FP16_TWO, FP16_THREE, FP16_FOUR, 0, 0);
        wait_result();
        check_result(FP16_TEN, "FP16: 2*3+4=10");

        // --- FP16->FP32 Accumulate (mode=100) ---
        $display("--- FP16->FP32 Acc Mode ---");
        // Initialize accumulator with 0, then acc += 1.0*1.0 = 1.0
        // First, set acc_en to load acc_in
        @(posedge clk);
        acc_en <= 1; acc_in <= FP32_ZERO;
        @(posedge clk);
        acc_en <= 0;
        repeat(2) @(posedge clk);

        // FP16: 1.0 * 1.0 + acc(0.0) -> result should be 1.0 in FP32
        apply_input(3'b100, FP16_ONE, FP16_ONE, 0, 1, FP32_ZERO);
        wait_result();
        check_result(FP32_ONE, "Acc16: 0+1*1=1");

        // FP16: 2.0 * 3.0 + acc(0.0) -> 6.0 in FP32
        // FP32 6.0 = 0x40C00000
        @(posedge clk);
        acc_en <= 1; acc_in <= FP32_ZERO;
        @(posedge clk);
        acc_en <= 0;
        repeat(2) @(posedge clk);

        apply_input(3'b100, FP16_TWO, FP16_THREE, 0, 1, FP32_ZERO);
        wait_result();
        check_result(32'h40C00000, "Acc16: 0+2*3=6");

        // Summary
        repeat(5) @(posedge clk);
        $display("=== Test Summary: %0d/%0d passed, %0d failed ===",
                 pass_count, test_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
