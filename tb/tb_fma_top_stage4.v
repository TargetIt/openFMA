// =============================================================================
// Testbench for Stage 4: Complete FMA Top with INT8 MAD
// =============================================================================
`timescale 1ns / 1ps

module tb_fma_top_stage4;

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

    fma_top_stage4 dut (
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

    // FP8 constants
    localparam FP8_ZERO = 8'h00;
    localparam FP8_ONE  = 8'h38;
    localparam FP8_TWO  = 8'h40;

    function [31:0] pack_fp8;
        input [7:0] v0, v1, v2, v3;
        pack_fp8 = {v3, v2, v1, v0};
    endfunction

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

    task check_result_16;
        input [15:0] expected;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (result[15:0] == expected) begin
                $display("PASS: %0s - result[15:0] = %h", test_name, result[15:0]);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - result[15:0] = %h (expected %h)", test_name, result[15:0], expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_overflow;
        input expected_ov;
        input [255:0] test_name;
        begin
            test_count = test_count + 1;
            if (overflow == expected_ov) begin
                $display("PASS: %0s - overflow = %b", test_name, overflow);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: %0s - overflow = %b (expected %b)", test_name, overflow, expected_ov);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("stage4_test.vcd");
        $dumpvars(0, tb_fma_top_stage4);

        rst_n = 0; valid_in = 0; mode = 0;
        A = 0; B = 0; C = 0; acc_en = 0; acc_in = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        $display("=== Stage 4: Complete FMA Top Testbench ===");

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

        // --- FP8x4 regression ---
        $display("--- FP8x4 Mode Regression ---");
        apply_input(3'b010,
            pack_fp8(FP8_ONE, FP8_ONE, FP8_ONE, FP8_ONE),
            pack_fp8(FP8_ONE, FP8_ONE, FP8_ONE, FP8_ONE),
            pack_fp8(FP8_ONE, FP8_ONE, FP8_ONE, FP8_ONE), 0, 0);
        wait_result();
        // 1*1+1 = 2 for each lane
        check_result(pack_fp8(FP8_TWO, FP8_TWO, FP8_TWO, FP8_TWO), "FP8x4: 1*1+1=2 all");

        // --- INT8 MAD Tests (mode=011) ---
        $display("--- INT8 MAD Mode Tests ---");

        // INT8: 2 * 3 + 4 = 10
        // A=2 (8'h02), B=3 (8'h03), C=4 (16'h0004)
        apply_input(3'b011, {24'b0, 8'h02}, {24'b0, 8'h03}, {16'b0, 16'h0004}, 0, 0);
        wait_result();
        check_result_16(16'h000A, "INT8: 2*3+4=10");

        // INT8: (-1) * 2 + 5 = 3
        // A=-1 (8'hFF), B=2 (8'h02), C=5 (16'h0005)
        apply_input(3'b011, {24'b0, 8'hFF}, {24'b0, 8'h02}, {16'b0, 16'h0005}, 0, 0);
        wait_result();
        check_result_16(16'h0003, "INT8: (-1)*2+5=3");

        // INT8: 10 * 10 + 0 = 100
        // A=10 (8'h0A), B=10 (8'h0A), C=0
        apply_input(3'b011, {24'b0, 8'h0A}, {24'b0, 8'h0A}, 32'h0, 0, 0);
        wait_result();
        check_result_16(16'h0064, "INT8: 10*10+0=100");

        // INT8: (-128) * 1 + 0 = -128
        // A=-128 (8'h80), B=1 (8'h01), C=0
        apply_input(3'b011, {24'b0, 8'h80}, {24'b0, 8'h01}, 32'h0, 0, 0);
        wait_result();
        check_result_16(16'hFF80, "INT8: (-128)*1+0=-128");

        // INT8: 0 * 0 + 0 = 0
        apply_input(3'b011, 32'h0, 32'h0, 32'h0, 0, 0);
        wait_result();
        check_result_16(16'h0000, "INT8: 0*0+0=0");

        // INT8: 127 * 127 + 0 = 16129
        // A=127 (8'h7F), B=127 (8'h7F), C=0
        apply_input(3'b011, {24'b0, 8'h7F}, {24'b0, 8'h7F}, 32'h0, 0, 0);
        wait_result();
        // 16129 = 0x3F01
        check_result_16(16'h3F01, "INT8: 127*127=16129");

        // --- Acc16 regression ---
        $display("--- Acc16 Mode Regression ---");
        @(posedge clk);
        acc_en <= 1; acc_in <= FP32_ZERO;
        @(posedge clk);
        acc_en <= 0;
        repeat(2) @(posedge clk);

        apply_input(3'b100, FP16_ONE, FP16_ONE, 0, 1, FP32_ZERO);
        wait_result();
        check_result(FP32_ONE, "Acc16: 0+1*1=1");

        // Summary
        repeat(5) @(posedge clk);
        $display("=== Test Summary: %0d/%0d passed, %0d failed ===",
                 pass_count, test_count, fail_count);
        if (fail_count == 0) $display("ALL TESTS PASSED");
        else $display("SOME TESTS FAILED");
        $finish;
    end

endmodule
