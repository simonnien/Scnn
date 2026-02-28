`timescale 1 ns / 1 ps
`celldefine
module SRAM_10x12 (
    Q,
    CLK,
    CEN,
    WEN,
    A,
    D,
    EMA
);

    output [9:0] Q;
    input CLK;
    input CEN;
    input WEN;
    input [4:0] A;
    input [9:0] D;
    input [2:0] EMA;

endmodule
