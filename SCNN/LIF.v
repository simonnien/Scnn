module LIF (
    input  wire        CLK,
    input  wire        en,
    //---------------------------
    input  wire [12:0] Weight,
    input  wire [ 9:0] addr,
    //---------------------------
    output wire        OFM_data,
    output wire [ 9:0] OFM_addr,
    //---------------------------
    input  wire [ 3:0] data_Beta,
    input  wire [11:0] data_Vth,
    input  wire [11:0] data_bias,
    //---------------------------
    input  wire        Leakage_en,
    input  wire        Spike_en,
    input  wire        MP_Reset


);


endmodule
