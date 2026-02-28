module CU (
    input  wire        CLK,
    input  wire        en,
    input  wire        Reset,       //work?
    //---------------------------IOU to CU
    input  wire [ 7:0] CMD,
    input  wire [ 9:0] Cin,
    input  wire [ 4:0] T,
    //---------------------------CU to (CONV, LIF, PL)
    output reg         CONV_en,
    output reg         CONV_trig,
    output reg         LIF_en,
    output reg         PL_en,
    output reg         PL_trig,
    //---------------------------CU to LIF
    output wire        Leakage_en,
    output wire        Spike_en,
    output wire        MP_Reset,
    //---------------------------
    input  wire        CMD_en,
    input  wire        buf_full,
    input  wire        next_Cin,
    output wire [10:0] reg_en
);


endmodule
