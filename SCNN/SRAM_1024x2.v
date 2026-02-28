`timescale 1 ns / 1 ps
`celldefine
module SRAM_1024x2 (
    Q,
    CLK,
    CEN,
    WEN,
    A,
    D,
    EMA
);

    output [1:0] Q;
    input CLK;
    input CEN;
    input WEN;
    input [10:0] A;
    input [1:0] D;
    input [2:0] EMA;

endmodule
