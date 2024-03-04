module SNN(
    //Input Port
    clk,
    rst_n,
    in_valid,
    Img,
    Kernel,
	Weight,
    Opt,

    //Output Port
    out_valid,
    out
    );

// IEEE floating point parameter
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
parameter inst_arch = 0;
parameter inst_faithful_round = 0;

input rst_n, clk, in_valid;
input [inst_sig_width+inst_exp_width:0] Img, Kernel, Weight;
input [1:0] Opt;

output reg	out_valid;
output reg [inst_sig_width+inst_exp_width:0] out;

reg [6:0] input_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) input_cnt <= 7'd0;
    else begin
       if (in_valid) input_cnt <= input_cnt + 1'b1;
       else          input_cnt <= 7'd0;
    end
end

reg [1:0] s_opt; // static opt
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) s_opt <= 2'b0;
    else begin
        if (in_valid && input_cnt == 7'd0) s_opt <= Opt;
        else                               s_opt <= s_opt;
    end
end

wire [31:0] pixel_out0;
wire [31:0] pixel_out1;
wire [31:0] pixel_out2;
wire [31:0] nxt_out0;
wire [31:0] nxt_out1;
wire [31:0] nxt_out2;
wire ready; 

Pixel_reg pr1(
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .Opt_0(s_opt[0]), .ready(ready), .reset_sig(out_valid),
    .pixel_in(Img),
    .pixel_out0(pixel_out0),
    .pixel_out1(pixel_out1),
    .pixel_out2(pixel_out2),
    .nxt_out0(nxt_out0),
    .nxt_out1(nxt_out1),
    .nxt_out2(nxt_out2)
);

wire [31:0] k00, k01, k02;
wire [31:0] k10, k11, k12;
wire [31:0] k20, k21, k22;
wire [31:0] nxt_k00, nxt_k01;
wire [31:0] nxt_k10, nxt_k11;
wire [31:0] nxt_k20, nxt_k21;

Kernel_reg kr1(
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .reset_sig(out_valid),
    .kernel_in(Kernel),
    .k00(k00), .k01(k01), .k02(k02),
    .k10(k10), .k11(k11), .k12(k12),
    .k20(k20), .k21(k21), .k22(k22),
    .nxt_k00(nxt_k00), .nxt_k01(nxt_k01),
    .nxt_k10(nxt_k10), .nxt_k11(nxt_k11),
    .nxt_k20(nxt_k20), .nxt_k21(nxt_k21)
);

wire [31:0] sum;

PE_array pa1(
    .clk(clk), .rst_n(rst_n), .PE_in_en(ready),
    .k00(k00), .k01(k01), .k02(k02),
    .k10(k10), .k11(k11), .k12(k12),
    .k20(k20), .k21(k21), .k22(k22),
    .nxt_k00(nxt_k00), .nxt_k01(nxt_k01),
    .nxt_k10(nxt_k10), .nxt_k11(nxt_k11),
    .nxt_k20(nxt_k20), .nxt_k21(nxt_k21),
    .pixel_in0(pixel_out0), .pixel_in1(pixel_out1), .pixel_in2(pixel_out2),
    .nxt_in0(nxt_out0), .nxt_in1(nxt_out1), .nxt_in2(nxt_out2),
    .sum(sum)
);

wire [31:0] conv_out;

Conv_reg cr1(
    .clk(clk), .rst_n(rst_n),
    .sum(sum), .conv_out(conv_out)
);

wire [31:0] max_pixel;
wire pool_ready;
Max_pooling mp1(
    .clk(clk), .rst_n(rst_n), .conv_out(conv_out), .reset_sig(out_valid),
    .pool_en(ready), .max_pixel(max_pixel), .pool_ready(pool_ready)
);

wire [31:0] fc_out;
wire fc_ready;

FC_normal fn1(
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .pool_ready(pool_ready), .reset_sig(out_valid),
    .max_pixel(max_pixel), .weight(Weight),
    .fc_out(fc_out), .fc_ready(fc_ready)
);

wire [31:0] act_out;
wire act_ready;
Activation act1(
    .clk(clk), .rst_n(rst_n), .Opt_1(s_opt[1]), .fc_ready(fc_ready),
    .fc_out(fc_out), .act_out(act_out), .act_ready(act_ready)
);

Distance dis1( 
    .clk(clk), .rst_n(rst_n), .act_ready(act_ready), .act_out(act_out),
    .out_valid(out_valid), .out(out)
);

endmodule


module Pixel_reg(
    clk, rst_n, in_valid, Opt_0, ready, reset_sig,
    pixel_in,
    pixel_out0,
    pixel_out1,
    pixel_out2,
    nxt_out0,
    nxt_out1,
    nxt_out2
);
input  clk, rst_n, in_valid, Opt_0, reset_sig; // Opt_0 = 1'b0 -> Replication Padding, Opt_0 = 1'b1 -> Zero Padding 
input  [31:0] pixel_in;
output reg [31:0] pixel_out0, pixel_out1, pixel_out2;
output reg [31:0] nxt_out0, nxt_out1, nxt_out2;
output reg ready;

reg [3:0] in_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) in_cnt <= 4'd0;
    else begin
        if (in_valid) in_cnt <= in_cnt + 1'b1; 
        else          in_cnt <= 4'd0;
    end
end

// image pixel
reg [31:0] p11, p12, p13, p14;
reg [31:0] p21, p22, p23, p24;
reg [31:0] p31, p32, p33, p34;
reg [31:0] p41, p42, p43, p44;

// padding boundary
reg [31:0] C0, C1, C2, C3; // corner
reg [31:0] U0, U1; // upper bound
reg [31:0] D0, D1; // lower bound
reg [31:0] L0, L1; // left bound
reg [31:0] R0, R1; // right bound

always @* begin
    if (Opt_0) begin
        C0 = 32'd0; C1 = 32'd0; C2 = 32'd0; C3 = 32'd0;
        U0 = 32'd0; U1 = 32'd0; D0 = 32'd0; D1 = 32'd0;
        L0 = 32'd0; L1 = 32'd0; R0 = 32'd0; R1 = 32'd0;
    end
    else begin
        C0 = p11; C1 = p14; C2 = p41; C3 = p44;
        U0 = p12; U1 = p13; D0 = p42; D1 = p43;
        L0 = p21; L1 = p31; R0 = p24; R1 = p34;
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        p11 <= 32'd0;  p12 <= 32'd0;  p13 <= 32'd0;  p14 <= 32'd0;
        p21 <= 32'd0;  p22 <= 32'd0;  p23 <= 32'd0;  p24 <= 32'd0;
        p31 <= 32'd0;  p32 <= 32'd0;  p33 <= 32'd0;  p34 <= 32'd0;
        p41 <= 32'd0;  p42 <= 32'd0;  p43 <= 32'd0;  p44 <= 32'd0;
    end
    else begin
        if (in_valid) begin
            case (in_cnt)
                4'd0 : p11 <= pixel_in;
                4'd1 : p12 <= pixel_in;
                4'd2 : p13 <= pixel_in;
                4'd3 : p14 <= pixel_in;
                4'd4 : p21 <= pixel_in;
                4'd5 : p22 <= pixel_in;
                4'd6 : p23 <= pixel_in;
                4'd7 : p24 <= pixel_in;
                4'd8 : p31 <= pixel_in;
                4'd9 : p32 <= pixel_in;
                4'd10: p33 <= pixel_in;
                4'd11: p34 <= pixel_in;
                4'd12: p41 <= pixel_in;
                4'd13: p42 <= pixel_in;
                4'd14: p43 <= pixel_in;
                4'd15: p44 <= pixel_in;
            endcase
        end
        else begin
            p11 <= p11;  p12 <= p12;  p13 <= p13;  p14 <= p14;
            p21 <= p21;  p22 <= p22;  p23 <= p23;  p24 <= p24;
            p31 <= p31;  p32 <= p32;  p33 <= p33;  p34 <= p34;
            p41 <= p41;  p42 <= p42;  p43 <= p43;  p44 <= p44;
        end
    end
end

reg [4:0] out_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_cnt <= 5'd0;
        ready   <= 1'b0;
    end
    else begin
        if (reset_sig) begin
            out_cnt <= 5'd0;
            ready   <= 1'b0;
        end
        else begin
            if (in_cnt < 4'd6 && !ready) begin // can start output data after in_cnt > 5
                out_cnt <= out_cnt;
                ready   <= ready;
            end
            else begin
                ready   <= 1'b1;
                if (out_cnt == 5'd18) out_cnt <= 5'd3;
                else                  out_cnt <= out_cnt + 1;
            end
        end
    end
end

always @* begin
    case (out_cnt)
        5'd1: begin
            pixel_out0 = C0; nxt_out0 = U0;
            pixel_out1 = C0; nxt_out1 = p12;
            pixel_out2 = L0; nxt_out2 = p22;
        end
        5'd2: begin
            pixel_out0 = C0;  nxt_out0 = U1; 
            pixel_out1 = p11; nxt_out1 = p13;
            pixel_out2 = p21; nxt_out2 = p23;  
        end
        5'd3: begin
            pixel_out0 = U0;  nxt_out0 = C1; 
            pixel_out1 = p12; nxt_out1 = p14;
            pixel_out2 = p22; nxt_out2 = p24;  
        end
        5'd4: begin
            pixel_out0 = U1;  nxt_out0 = C1; 
            pixel_out1 = p13; nxt_out1 = C1;
            pixel_out2 = p23; nxt_out2 = R0;  
        end
        5'd5: begin
            pixel_out0 = C1;  nxt_out0 = C0; 
            pixel_out1 = p14; nxt_out1 = L0;
            pixel_out2 = p24; nxt_out2 = L1;  
        end
        5'd6: begin
            pixel_out0 = C1; nxt_out0 = p11; 
            pixel_out1 = C1; nxt_out1 = p21;
            pixel_out2 = R0; nxt_out2 = p31;  
        end
        5'd7: begin
            pixel_out0 = p12; nxt_out0 = p14; 
            pixel_out1 = p22; nxt_out1 = p24;
            pixel_out2 = p32; nxt_out2 = p34;  
        end
        5'd8: begin
            pixel_out0 = p13; nxt_out0 = C1; 
            pixel_out1 = p23; nxt_out1 = R0;
            pixel_out2 = p33; nxt_out2 = R1;  
        end
        5'd9: begin
            pixel_out0 = p14; nxt_out0 = L0; 
            pixel_out1 = p24; nxt_out1 = L1;
            pixel_out2 = p34; nxt_out2 = C2;  
        end
        5'd10: begin
            pixel_out0 = C1; nxt_out0 = p21; 
            pixel_out1 = R0; nxt_out1 = p31;
            pixel_out2 = R1; nxt_out2 = p41;  
        end
        5'd11: begin
            pixel_out0 = p22; nxt_out0 = p24; 
            pixel_out1 = p32; nxt_out1 = p34;
            pixel_out2 = p42; nxt_out2 = p44;  
        end
        5'd12: begin
            pixel_out0 = p23; nxt_out0 = R0; 
            pixel_out1 = p33; nxt_out1 = R1;
            pixel_out2 = p43; nxt_out2 = C3;  
        end
        5'd13: begin
            pixel_out0 = p24; nxt_out0 = L1; 
            pixel_out1 = p34; nxt_out1 = C2;
            pixel_out2 = p44; nxt_out2 = C2;  
        end
        5'd14: begin
            pixel_out0 = R0; nxt_out0 = p31; 
            pixel_out1 = R1; nxt_out1 = p41;
            pixel_out2 = C3; nxt_out2 = C2;  
        end
        5'd15: begin
            pixel_out0 = p32; nxt_out0 = p34; 
            pixel_out1 = p42; nxt_out1 = p44;
            pixel_out2 = D0;  nxt_out2 = C3;  
        end 
        5'd16: begin
            pixel_out0 = p33; nxt_out0 = R1; 
            pixel_out1 = p43; nxt_out1 = C3;
            pixel_out2 = D1;  nxt_out2 = C3;  
        end
        5'd17: begin
            pixel_out0 = p34; nxt_out0 = C0; 
            pixel_out1 = p44; nxt_out1 = C0;
            pixel_out2 = C3;  nxt_out2 = L0;  
        end
        5'd18: begin
            pixel_out0 = R1; nxt_out0 = C0; 
            pixel_out1 = C3; nxt_out1 = p11;
            pixel_out2 = C3; nxt_out2 = p21;  
        end
        default: begin
            pixel_out0 = 32'd0; nxt_out0 = 32'd0;
            pixel_out1 = 32'd0; nxt_out1 = 32'd0;
            pixel_out2 = 32'd0; nxt_out2 = 32'd0;  
        end
    endcase
end

endmodule

module Kernel_reg(
    clk, rst_n, in_valid, reset_sig,
    kernel_in,
    k00, k01, k02,
    k10, k11, k12,
    k20, k21, k22,
    nxt_k00, nxt_k01,
    nxt_k10, nxt_k11,
    nxt_k20, nxt_k21
);
input  clk, rst_n, in_valid, reset_sig; // Opt_0 = 1'b0 -> Replication Padding, Opt_0 = 1'b1 -> Zero Padding 
input  [31:0] kernel_in;
output reg [31:0] k00, k01, k02;
output reg [31:0] k10, k11, k12;
output reg [31:0] k20, k21, k22;
output reg [31:0] nxt_k00, nxt_k01;
output reg [31:0] nxt_k10, nxt_k11;
output reg [31:0] nxt_k20, nxt_k21;

reg [4:0] in_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) in_cnt <= 5'd0;
    else begin
        if (in_valid) begin 
            if (in_cnt < 5'd27) in_cnt <= in_cnt + 1'b1;
            else                in_cnt <= in_cnt;
        end 
        else          in_cnt <= 5'd0;
    end
end

// kernel
reg [31:0] k00_0, k01_0, k02_0;
reg [31:0] k10_0, k11_0, k12_0;
reg [31:0] k20_0, k21_0, k22_0;

reg [31:0] k00_1, k01_1, k02_1;
reg [31:0] k10_1, k11_1, k12_1;
reg [31:0] k20_1, k21_1, k22_1;

reg [31:0] k00_2, k01_2, k02_2;
reg [31:0] k10_2, k11_2, k12_2;
reg [31:0] k20_2, k21_2, k22_2;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        k00_0 <= 32'd0; k01_0 <= 32'd0; k02_0 <= 32'd0;
        k10_0 <= 32'd0; k11_0 <= 32'd0; k12_0 <= 32'd0;
        k20_0 <= 32'd0; k21_0 <= 32'd0; k22_0 <= 32'd0;
        
        k00_1 <= 32'd0; k01_1 <= 32'd0; k02_1 <= 32'd0;
        k10_1 <= 32'd0; k11_1 <= 32'd0; k12_1 <= 32'd0;
        k20_1 <= 32'd0; k21_1 <= 32'd0; k22_1 <= 32'd0;
        
        k00_2 <= 32'd0; k01_2 <= 32'd0; k02_2 <= 32'd0;
        k10_2 <= 32'd0; k11_2 <= 32'd0; k12_2 <= 32'd0;
        k20_2 <= 32'd0; k21_2 <= 32'd0; k22_2 <= 32'd0;
    end
    else begin
        if (in_valid) begin
            case (in_cnt)
                5'd0 : k00_0 <= kernel_in;
                5'd1 : k01_0 <= kernel_in;
                5'd2 : k02_0 <= kernel_in;
                5'd3 : k10_0 <= kernel_in;
                5'd4 : k11_0 <= kernel_in;
                5'd5 : k12_0 <= kernel_in;
                5'd6 : k20_0 <= kernel_in;
                5'd7 : k21_0 <= kernel_in;
                5'd8 : k22_0 <= kernel_in;
                5'd9 : k00_1 <= kernel_in;
                5'd10: k01_1 <= kernel_in;
                5'd11: k02_1 <= kernel_in;
                5'd12: k10_1 <= kernel_in;
                5'd13: k11_1 <= kernel_in;
                5'd14: k12_1 <= kernel_in;
                5'd15: k20_1 <= kernel_in;
                5'd16: k21_1 <= kernel_in;
                5'd17: k22_1 <= kernel_in;
                5'd18: k00_2 <= kernel_in;
                5'd19: k01_2 <= kernel_in;
                5'd20: k02_2 <= kernel_in;
                5'd21: k10_2 <= kernel_in;
                5'd22: k11_2 <= kernel_in;
                5'd23: k12_2 <= kernel_in;
                5'd24: k20_2 <= kernel_in;
                5'd25: k21_2 <= kernel_in;
                5'd26: k22_2 <= kernel_in;
                default: begin
                    k00_0 <= k00_0; k01_0 <= k01_0; k02_0 <= k02_0;
                    k10_0 <= k10_0; k11_0 <= k11_0; k12_0 <= k12_0;
                    k20_0 <= k20_0; k21_0 <= k21_0; k22_0 <= k22_0;
                    
                    k00_1 <= k00_1; k01_1 <= k01_1; k02_1 <= k02_1;
                    k10_1 <= k10_1; k11_1 <= k11_1; k12_1 <= k12_1;
                    k20_1 <= k20_1; k21_1 <= k21_1; k22_1 <= k22_1;
                    
                    k00_2 <= k00_2; k01_2 <= k01_2; k02_2 <= k02_2;
                    k10_2 <= k10_2; k11_2 <= k11_2; k12_2 <= k12_2;
                    k20_2 <= k20_2; k21_2 <= k21_2; k22_2 <= k22_2;
                end
            endcase
        end
        else begin
            k00_0 <= k00_0; k01_0 <= k01_0; k02_0 <= k02_0;
            k10_0 <= k10_0; k11_0 <= k11_0; k12_0 <= k12_0;
            k20_0 <= k20_0; k21_0 <= k21_0; k22_0 <= k22_0;
            
            k00_1 <= k00_1; k01_1 <= k01_1; k02_1 <= k02_1;
            k10_1 <= k10_1; k11_1 <= k11_1; k12_1 <= k12_1;
            k20_1 <= k20_1; k21_1 <= k21_1; k22_1 <= k22_1;
            
            k00_2 <= k00_2; k01_2 <= k01_2; k02_2 <= k02_2;
            k10_2 <= k10_2; k11_2 <= k11_2; k12_2 <= k12_2;
            k20_2 <= k20_2; k21_2 <= k21_2; k22_2 <= k22_2;
        end
    end
end

reg [5:0] out_cnt;
reg ready;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_cnt <= 6'd0;
        ready   <= 1'b0;
    end
    else begin
        if (reset_sig) begin
            out_cnt <= 6'd0;
            ready   <= 1'b0;
        end
        else begin
            if (in_cnt < 4'd6 && !ready) begin // can start output data after in_cnt > 5
                out_cnt <= out_cnt;
                ready   <= ready;
            end
            else begin
                ready   <= 1'b1;
                if (out_cnt == 6'd50) out_cnt <= 6'd3;
                else                  out_cnt <= out_cnt + 1;
            end
        end 
    end
end

always @* begin
    case (out_cnt)
        6'd1, 6'd2, 6'd3, 6'd4, 6'd5, 6'd6, 6'd7, 6'd8, 6'd9, 6'd10,
        6'd11, 6'd12, 6'd13, 6'd14, 6'd15, 6'd16, 6'd17, 6'd18: begin
            k00 = k00_0; k01 = k01_0; k02 = k02_0;
            k10 = k10_0; k11 = k11_0; k12 = k12_0;
            k20 = k20_0; k21 = k21_0; k22 = k22_0;
            nxt_k00 = k00_1; nxt_k01 = k01_1;
            nxt_k10 = k10_1; nxt_k11 = k11_1;
            nxt_k20 = k20_1; nxt_k21 = k21_1;
        end
        6'd19, 6'd20, 6'd21, 6'd22, 6'd23, 6'd24, 6'd25, 6'd26, 6'd27, 6'd28, 
        6'd29, 6'd30, 6'd31, 6'd32, 6'd33, 6'd34: begin
            k00 = k00_1; k01 = k01_1; k02 = k02_1;
            k10 = k10_1; k11 = k11_1; k12 = k12_1;
            k20 = k20_1; k21 = k21_1; k22 = k22_1;
            nxt_k00 = k00_2; nxt_k01 = k01_2;
            nxt_k10 = k10_2; nxt_k11 = k11_2;
            nxt_k20 = k20_2; nxt_k21 = k21_2;
        end
        6'd35, 6'd36, 6'd37, 6'd38, 6'd39, 6'd40, 6'd41, 6'd42, 6'd43, 6'd44,
        6'd45, 6'd46, 6'd47, 6'd48, 6'd49, 6'd50: begin
            k00 = k00_2; k01 = k01_2; k02 = k02_2;
            k10 = k10_2; k11 = k11_2; k12 = k12_2;
            k20 = k20_2; k21 = k21_2; k22 = k22_2;
            nxt_k00 = k00_0; nxt_k01 = k01_0;
            nxt_k10 = k10_0; nxt_k11 = k11_0;
            nxt_k20 = k20_0; nxt_k21 = k21_0;
        end
        default: begin
            k00 = 32'd0; k01 = 32'd0; k02 = 32'd0;
            k10 = 32'd0; k11 = 32'd0; k12 = 32'd0;
            k20 = 32'd0; k21 = 32'd0; k22 = 32'd0;
            nxt_k00 = 32'd0; nxt_k01 = 32'd0;
            nxt_k10 = 32'd0; nxt_k11 = 32'd0;
            nxt_k20 = 32'd0; nxt_k21 = 32'd0;
        end
    endcase
end

endmodule

// Processing Element
module PE_array(
    clk, rst_n, PE_in_en,
    k00, k01, k02,
    k10, k11, k12,
    k20, k21, k22,
    nxt_k00, nxt_k01,
    nxt_k10, nxt_k11,
    nxt_k20, nxt_k21,
    pixel_in0, pixel_in1, pixel_in2,
    nxt_in0, nxt_in1, nxt_in2,
    sum
);
input clk, rst_n, PE_in_en;
input [31:0] k00, k01, k02;
input [31:0] k10, k11, k12;
input [31:0] k20, k21, k22;
input [31:0] nxt_k00, nxt_k01;
input [31:0] nxt_k10, nxt_k11;
input [31:0] nxt_k20, nxt_k21;
input [31:0] pixel_in0, pixel_in1, pixel_in2;
input [31:0] nxt_in0, nxt_in1, nxt_in2;
output [31:0] sum;

reg [4:0] pe_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) pe_cnt <= 5'd0;
    else begin
        if (PE_in_en) begin
            if (pe_cnt == 5'd17) pe_cnt <= 5'd2;
            else                 pe_cnt <= pe_cnt + 1;
        end
        else pe_cnt <= 5'd0;
    end
end

reg [31:0] kernel_00, kernel_01;
reg [31:0] kernel_10, kernel_11;
reg [31:0] kernel_20, kernel_21;
always @* begin
    case (pe_cnt)
        6'd0, 6'd1, 6'd2, 6'd3, 6'd4, 6'd5, 6'd6, 6'd7, 6'd8, 6'd9, 
        6'd10, 6'd11, 6'd12, 6'd13, 6'd14, 6'd15: begin
            kernel_00 = k00; kernel_01 = k01; 
            kernel_10 = k10; kernel_11 = k11; 
            kernel_20 = k20; kernel_21 = k21; 
        end
        6'd16: begin
            kernel_00 = nxt_k00; kernel_01 = k01;
            kernel_10 = nxt_k10; kernel_11 = k11; 
            kernel_20 = nxt_k20; kernel_21 = k21;
        end
        6'd17: begin
            kernel_00 = nxt_k00; kernel_01 = nxt_k01;
            kernel_10 = nxt_k10; kernel_11 = nxt_k11;
            kernel_20 = nxt_k20; kernel_21 = nxt_k21;
        end
        default: begin
            kernel_00 = 32'd0; kernel_01 = 32'd0;
            kernel_10 = 32'd0; kernel_11 = 32'd0;
            kernel_20 = 32'd0; kernel_21 = 32'd0;
        end
    endcase
end

reg [31:0] pixel_00, pixel_10, pixel_20;
always @* begin
    if (pe_cnt <= 5'd3) begin
        pixel_00 = pixel_in0;
        pixel_10 = pixel_in1;
        pixel_20 = pixel_in2;    
    end
    else begin
        if (~pe_cnt[1]) begin // 00 or 01
            pixel_00 = nxt_in0;
            pixel_10 = nxt_in1;
            pixel_20 = nxt_in2; 
        end
        else begin
            pixel_00 = pixel_in0;
            pixel_10 = pixel_in1;
            pixel_20 = pixel_in2;         
        end
    end
end

reg [31:0] pixel_01, pixel_11, pixel_21;
always @* begin
    if (pe_cnt <= 5'd3) begin
        pixel_01 = pixel_in0;
        pixel_11 = pixel_in1;
        pixel_21 = pixel_in2;    
    end
    else begin
        if (pe_cnt[1:0] == 2'b01) begin
            pixel_01 = nxt_in0;
            pixel_11 = nxt_in1;
            pixel_21 = nxt_in2; 
        end
        else begin
            pixel_01 = pixel_in0;
            pixel_11 = pixel_in1;
            pixel_21 = pixel_in2;         
        end
    end
end

wire [31:0] bias_L1 [0:2], bias_L2 [0:2], bias_L3 [0:2];
reg [31:0] p00, p01, p02;
reg [31:0] p10, p11, p12;
reg [31:0] p20, p21, p22;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        p00 <= 32'd0; p01 <= 32'd0; p02 <= 32'd0;
        p10 <= 32'd0; p11 <= 32'd0; p12 <= 32'd0;
        p20 <= 32'd0; p21 <= 32'd0; p22 <= 32'd0;
    end
    else begin
        p00 <= bias_L1[0]; p01 <= bias_L2[0]; p02 <= bias_L3[0];
        p10 <= bias_L1[1]; p11 <= bias_L2[1]; p12 <= bias_L3[1];
        p20 <= bias_L1[2]; p21 <= bias_L2[2]; p22 <= bias_L3[2];
    end
end

DW_fp_mult_inst PE00( .pixel(pixel_00), .kernel(kernel_00), .out(bias_L1[0]) );
DW_fp_mult_inst PE10( .pixel(pixel_10), .kernel(kernel_10), .out(bias_L1[1]) );
DW_fp_mult_inst PE20( .pixel(pixel_20), .kernel(kernel_20), .out(bias_L1[2]) );

DW_fp_mac_inst PE01( .pixel(pixel_01), .kernel(kernel_01), .bias(p00), .out(bias_L2[0]) );
DW_fp_mac_inst PE11( .pixel(pixel_11), .kernel(kernel_11), .bias(p10), .out(bias_L2[1]) );
DW_fp_mac_inst PE21( .pixel(pixel_21), .kernel(kernel_21), .bias(p20), .out(bias_L2[2]) );

DW_fp_mac_inst PE02( .pixel(pixel_in0), .kernel(k02), .bias(p01), .out(bias_L3[0]) );
DW_fp_mac_inst PE12( .pixel(pixel_in1), .kernel(k12), .bias(p11), .out(bias_L3[1]) );
DW_fp_mac_inst PE22( .pixel(pixel_in2), .kernel(k22), .bias(p21), .out(bias_L3[2]) );

DW_fp_sum3_inst sum3( .a(p02), .b(p12), .c(p22), .d(sum) );

endmodule

// Can resuce to fp_sum2
module Conv_reg(
    clk, rst_n,
    sum, conv_out
);

input clk, rst_n;
input [31:0] sum;
output [31:0] conv_out;

reg [31:0] c00_0, c01_0, c02_0, c03_0;
reg [31:0] c10_0, c11_0, c12_0, c13_0;
reg [31:0] c20_0, c21_0, c22_0, c23_0;
reg [31:0] c30_0, c31_0, c32_0, c33_0;

reg [31:0] c00_1, c01_1, c02_1, c03_1;
reg [31:0] c10_1, c11_1, c12_1, c13_1;
reg [31:0] c20_1, c21_1, c22_1, c23_1;
reg [31:0] c30_1, c31_1, c32_1, c33_1;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        c00_0 <= 32'd0; c01_0 <= 32'd0; c02_0 <= 32'd0; c03_0 <= 32'd0;
        c10_0 <= 32'd0; c11_0 <= 32'd0; c12_0 <= 32'd0; c13_0 <= 32'd0;
        c20_0 <= 32'd0; c21_0 <= 32'd0; c22_0 <= 32'd0; c23_0 <= 32'd0;
        c30_0 <= 32'd0; c31_0 <= 32'd0; c32_0 <= 32'd0; c33_0 <= 32'd0;
        
        c00_1 <= 32'd0; c01_1 <= 32'd0; c02_1 <= 32'd0; c03_1 <= 32'd0;
        c10_1 <= 32'd0; c11_1 <= 32'd0; c12_1 <= 32'd0; c13_1 <= 32'd0;
        c20_1 <= 32'd0; c21_1 <= 32'd0; c22_1 <= 32'd0; c23_1 <= 32'd0;
        c30_1 <= 32'd0; c31_1 <= 32'd0; c32_1 <= 32'd0; c33_1 <= 32'd0;
    end
    else begin
        c00_0 <= c01_0; c01_0 <= c02_0; c02_0 <= c03_0; c03_0 <= c10_0;
        c10_0 <= c11_0; c11_0 <= c12_0; c12_0 <= c13_0; c13_0 <= c20_0;
        c20_0 <= c21_0; c21_0 <= c22_0; c22_0 <= c23_0; c23_0 <= c30_0;
        c30_0 <= c31_0; c31_0 <= c32_0; c32_0 <= c33_0; c33_0 <= c00_1;
        
        c00_1 <= c01_1; c01_1 <= c02_1; c02_1 <= c03_1; c03_1 <= c10_1;
        c10_1 <= c11_1; c11_1 <= c12_1; c12_1 <= c13_1; c13_1 <= c20_1;
        c20_1 <= c21_1; c21_1 <= c22_1; c22_1 <= c23_1; c23_1 <= c30_1;
        c30_1 <= c31_1; c31_1 <= c32_1; c32_1 <= c33_1; c33_1 <= sum;
    end
end

DW_fp_sum3_inst sum_conv( .a(c00_0), .b(c00_1), .c(sum), .d(conv_out) );

endmodule

module Max_pooling(
    clk, rst_n, conv_out, pool_en, reset_sig,
    max_pixel, pool_ready
);
input clk, rst_n, pool_en, reset_sig;
input [31:0] conv_out;
output reg [31:0] max_pixel;
output reg pool_ready;

reg [6:0] pool_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) pool_cnt <= 7'd5;
    else begin
        if (reset_sig) begin
            pool_cnt <= 7'd5;
        end
        else begin
            if (pool_en) pool_cnt <= pool_cnt + 7'd1;
            else         pool_cnt <= pool_cnt;
        end
    end
end

wire [31:0] z1; // output from fp_cmp
reg [31:0] max_pre_1, max_pre_2;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        max_pre_1 <= 32'd0; 
    end
    else begin
        if (reset_sig) begin
            max_pre_1 <= 32'd0; 
        end
        else begin
            if (pool_cnt[2:0] == 3'd1) max_pre_1 <= z1;
            else                       max_pre_1 <= max_pre_1;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        max_pre_2 <= 32'd0; 
    end
    else begin
        if (reset_sig) begin
            max_pre_2 <= 32'd0; 
        end
        else begin
            if (pool_cnt[2:0] == 3'd3) max_pre_2 <= z1;
            else                       max_pre_2 <= max_pre_2;
        end
    end
end

reg [31:0] max_in;
reg [31:0] b_in;
always @* begin
    case (pool_cnt[1:0])
        2'b00: b_in = max_pre_1;
        2'b01: b_in = max_pixel;
        2'b10: b_in = max_pre_2;
        2'b11: b_in = max_pixel;
    endcase
end

always @* begin
    if (pool_cnt[2] | pool_cnt[0]) max_in = z1;
    else                           max_in = conv_out;
end

always @(posedge clk or negedge rst_n) begin 
    if (!rst_n) max_pixel <= 32'd0;
    else        max_pixel <= max_in;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) pool_ready <= 1'b0;
    else begin
        if( ( (pool_cnt == 7'd45 || pool_cnt == 7'd47) || (pool_cnt == 7'd53 || pool_cnt == 7'd55) ) ||
            ( (pool_cnt == 7'd93 || pool_cnt == 7'd95) || (pool_cnt == 7'd101 || pool_cnt == 7'd103) ) )
            pool_ready <= 1'b1;
        else 
            pool_ready <= 1'b0;
    end
end

wire [31:0] z0; // useless signal
DW_fp_cmp_inst cmp1( .inst_a(conv_out), .inst_b(b_in), .z0_inst(z0), .z1_inst(z1) );

endmodule

module FC_normal(
    clk, rst_n, in_valid, pool_ready, reset_sig,
    max_pixel, weight,
    fc_out, fc_ready
);
input clk, rst_n, in_valid, pool_ready, reset_sig;
input [31:0] max_pixel, weight;
output reg fc_ready;
output [31:0] fc_out;

reg [1:0] in_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) in_cnt <= 2'd0;
    else begin
        if (in_valid) in_cnt <= in_cnt + 2'b1; 
        else          in_cnt <= 2'd0;
    end
end

reg [31:0] w00, w01, w10, w11;
reg finish_store;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) finish_store <= 1'b1;
    else begin
        if (reset_sig) begin
            finish_store <= 1'b1;
        end
        else begin
            if (&in_cnt) finish_store <= 1'b0;
            else         finish_store <= finish_store;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        w00 <= 32'd0; 
        w01 <= 32'd0; 
        w10 <= 32'd0;
        w11 <= 32'd0;
    end
    else begin
        if (in_valid && finish_store) begin
            w00 <= w01; 
            w01 <= w10; 
            w10 <= w11;
            w11 <= weight;
        end 
        else begin
            w00 <= w00; 
            w01 <= w01; 
            w10 <= w10;
            w11 <= w11;
        end
    end
end

reg [31:0] n3_weight;
reg [2:0] pixel_cnt;
always @* begin
    case (pixel_cnt[1:0])
        2'b00: n3_weight = w00;
        2'b01: n3_weight = w01;
        2'b10: n3_weight = w10;
        2'b11: n3_weight = w11;
    endcase
end

wire [31:0] n3_in;
reg [31:0] n0, n1, n2, n3;
reg pre_pool_ready;
reg pool_ready_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        pre_pool_ready <= 1'b0;
    else begin
        if (reset_sig) begin
            pre_pool_ready <= 1'b0;
        end
        else begin
            if (pool_ready_cnt && !pool_ready) pre_pool_ready <= 1'b1;
            else                               pre_pool_ready <= pool_ready_cnt;
        end
    end        
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        n0 <= 32'd0;
        n1 <= 32'd0;
        n2 <= 32'd0;
        n3 <= 32'd0;
        pixel_cnt <= 3'd0;
    end
    else begin
        if (reset_sig) begin
            n0 <= 32'd0;
            n1 <= 32'd0;
            n2 <= 32'd0;
            n3 <= 32'd0;
            pixel_cnt <= 3'd0;
        end
        else begin
            if (pre_pool_ready || fc_ready) begin
                n0 <= n1;
                n1 <= n2;
                n2 <= n3;
                n3 <= n3_in;
                pixel_cnt <= pixel_cnt + 3'd1;
            end
            else begin
                n0 <= n0;
                n1 <= n1;
                n2 <= n2;
                n3 <= n3;
                pixel_cnt <= pixel_cnt;
            end
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fc_ready <= 1'b0;
    end
    else begin
        if ( (pixel_cnt == 3'd3 && pre_pool_ready) || 
             (pixel_cnt >= 3'd4 && pixel_cnt < 3'd7)) fc_ready <= 1'b1;
        else                                          fc_ready <= 1'b0;
    end
end

wire [31:0] X_min, X_max;
reg [31:0] s_X_min, s_X_max;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        s_X_min <= 32'd0;
        s_X_max <= 32'd0;
    end
    else begin
        if (reset_sig) begin
            s_X_min <= 32'd0;
            s_X_max <= 32'd0;
        end
        else begin
            if (pixel_cnt == 3'd4) begin
                s_X_min <= X_min;
                s_X_max <= X_max;
            end
            else begin
                s_X_min <= s_X_min;
                s_X_max <= s_X_max;
            end    
        end
    end
end

reg [31:0] p0, p1;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        p0 <= 32'b0;
        p1 <= 32'd0;
        pool_ready_cnt <= 1'b0;
    end
    else begin
        if (reset_sig) begin
            p0 <= 32'b0;
            p1 <= 32'd0;
            pool_ready_cnt <= 1'b0;
        end 
        else begin
            if (pool_ready) begin
                p0 <= p1;
                p1 <= max_pixel;     
                pool_ready_cnt <= ~pool_ready_cnt;
            end 
            else begin
                p0 <= p0;
                p1 <= p1;
                pool_ready_cnt <= pool_ready_cnt;
            end                   
        end
    end
end

reg [31:0] a_in, b_in, c_in, d_in;
always @* begin
    case(pixel_cnt[1:0])
        2'b00: begin
            a_in = p1; 
            b_in = w00;
            c_in = max_pixel;
            d_in = w10;
        end
        2'b01: begin
            a_in = p0; 
            b_in = w01;
            c_in = p1;
            d_in = w11;
        end
        2'b10: begin
            a_in = p1; 
            b_in = w00;
            c_in = max_pixel;
            d_in = w10;
        end
        2'b11: begin
            a_in = p0; 
            b_in = w01;
            c_in = p1;
            d_in = w11;
        end
    endcase
end 

DW_fp_dp2_inst dp1( .inst_a(a_in), .inst_b(b_in), .inst_c(c_in), .inst_d(d_in), .z_inst(n3_in) );

wire [31:0] z0 [0:1]; // useless
wire [31:0] max_x, max_y, min_x, min_y;
DW_fp_cmp_inst cmp1( .inst_a(n0), .inst_b(n1), .z0_inst(min_x), .z1_inst(max_x) );
DW_fp_cmp_inst cmp2( .inst_a(n2), .inst_b(n3), .z0_inst(min_y), .z1_inst(max_y) );
DW_fp_cmp_inst cmp3( .inst_a(max_x), .inst_b(max_y), .z0_inst(z0[0]), .z1_inst(X_max) );
DW_fp_cmp_inst cmp4( .inst_a(min_x), .inst_b(min_y), .z0_inst(X_min), .z1_inst(z0[1]) );

wire [31:0] xmax_in = (pixel_cnt == 3'd4)? X_max : s_X_max;
wire [31:0] xmin_in = (pixel_cnt == 3'd4)? X_min : s_X_min;

Normalization normal1(.x(n0), .xmin(xmin_in), .xmax(xmax_in), .x_scale(fc_out));

endmodule

module Normalization(
    x, xmin, xmax,
    x_scale
);
input [31:0] x, xmin, xmax;
output [31:0] x_scale;

wire [31:0] up, down;

DW_fp_sub_inst sub1( .inst_a(x), .inst_b(xmin), .z_inst(up) );
DW_fp_sub_inst sub2( .inst_a(xmax), .inst_b(xmin), .z_inst(down) );
DW_fp_div_inst div1(.inst_a(up), .inst_b(down), .z_inst(x_scale));

endmodule

module Activation(
    clk, rst_n, Opt_1, fc_ready,
    fc_out, act_out, act_ready
);
input clk, rst_n, Opt_1, fc_ready;
input [31:0] fc_out;
output [31:0] act_out;
output reg act_ready;

wire [31:0] exp_p, exp_n; // positive, negative
wire [31:0] tanh_up, tanh_down;
wire [31:0] sig_up, sig_down;
assign sig_up = {1'b0, 8'h7F, 23'd0}; // number "1" in IEEE 754 expression

reg [31:0] fc_in;
reg act_ready_tmp1, act_ready_tmp2;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fc_in        <= 32'd0;
        act_ready     <= 1'b0;
        act_ready_tmp1 <= 1'b0;
        act_ready_tmp2 <= 1'b0;
    end
    else begin
        fc_in          <= fc_out;
        act_ready_tmp2 <= fc_ready;
        act_ready_tmp1 <= act_ready_tmp2;
        act_ready      <= act_ready_tmp1;
    end
end

DW_fp_exp_inst expp( .inst_a(fc_in), .z_inst(exp_p) );
DW_fp_exp_inst expn( .inst_a({~fc_in[31], fc_in[30:0]}), .z_inst(exp_n) );

reg [31:0] exp_pp, exp_nn;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        exp_pp <= 32'd0;
        exp_nn <= 1'b0;
    end
    else begin
        exp_pp <= exp_p;
        exp_nn <= exp_n;
    end
end

DW_fp_add_inst add1( .inst_a(exp_pp), .inst_b(exp_nn), .z_inst(tanh_down) );
DW_fp_add_inst add2( .inst_a(sig_up), .inst_b(exp_nn), .z_inst(sig_down) );
DW_fp_sub_inst sub1( .inst_a(exp_pp), .inst_b(exp_nn), .z_inst(tanh_up) );

reg [31:0] div_up, div_down;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        div_up   <= 32'd0;
        div_down <= 1'b0;
    end
    else begin
        if (Opt_1) begin
            div_up   <= tanh_up;
            div_down <= tanh_down;
        end
        else begin
            div_up   <= sig_up;
            div_down <= sig_down;
        end
    end
end

DW_fp_div_inst div1( .inst_a(div_up), .inst_b(div_down), .z_inst(act_out) );

endmodule

module Distance( 
    clk, rst_n, act_ready, act_out,
    out_valid, out
);
input clk, rst_n, act_ready;
input [31:0] act_out;
output reg [31:0] out;
output reg out_valid;

reg [2:0] pixel_cnt;
reg [31:0] d0, d1, d2, d3, psum;
wire [31:0] sub_out, add_out;
reg pre_act_ready;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        d0 <= 32'd0;
        d1 <= 32'd0;
        d2 <= 32'd0;
        d3 <= 32'd0;
        psum <= 32'd0;
        pixel_cnt <= 3'd0;
    end
    else begin
        if (out_valid) begin
            d0 <= 32'd0;
            d1 <= 32'd0;
            d2 <= 32'd0;
            d3 <= 32'd0;
            psum <= 32'd0;
            pixel_cnt <= 3'd0;
        end
        else begin
            if (pre_act_ready) begin
                d0 <= d1;
                d1 <= d2;
                d2 <= d3;
                d3 <= sub_out;
                psum <= add_out;
                pixel_cnt <= pixel_cnt + 3'd1;
            end
            else begin
                d0 <= d0;
                d1 <= d1;
                d2 <= d2;
                d3 <= d3;
                pixel_cnt <= pixel_cnt;
                if (pixel_cnt == 3'd4) psum <= 32'd0;
                else                   psum <= psum;
            end
        end
    end
end

reg ready_to_output;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ready_to_output <= 1'b0;
        pre_act_ready     <= 1'b0;
    end
    else begin
        if (out_valid) begin
            ready_to_output <= 1'b0;
            pre_act_ready     <= 1'b0;
        end
        else begin
            pre_act_ready     <= act_ready;
            
            if (pixel_cnt == 3'd7)
                ready_to_output <= 1'b1;
            else
                ready_to_output <= 1'b0;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_valid <= 1'b0;
        out <= 32'd0;
    end
    else begin
        if (ready_to_output) begin
            out_valid <= 1'b1;
            out <= psum;
        end
        else begin
            out_valid <= 1'b0;
            out <= 32'd0;
        end
    end
end

reg [31:0] act_in;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        act_in <= 32'd0;
    end
    else begin
        act_in <= act_out;
    end
end

DW_fp_sub_inst sub1( .inst_a(act_in), .inst_b(d0), .z_inst(sub_out) );
DW_fp_add_inst add1( .inst_a({1'b0, sub_out[30:0]}), .inst_b(psum), .z_inst(add_out) );

endmodule

// z = a*b + c*d
module DW_fp_dp2_inst( inst_a, inst_b, inst_c, inst_d, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
input [inst_sig_width+inst_exp_width : 0] inst_c;
input [inst_sig_width+inst_exp_width : 0] inst_d;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_dp2
DW_fp_dp2 #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch_type)
U1 (
.a(inst_a),
.b(inst_b),
.c(inst_c),
.d(inst_d),
.rnd(inst_rnd),
.z(z_inst),
.status(status_inst) );
endmodule

module DW_fp_add_inst( inst_a, inst_b, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_add
DW_fp_add #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
U1 ( .a(inst_a), .b(inst_b), .rnd(inst_rnd), .z(z_inst), .status(status_inst) );
endmodule

module DW_fp_exp_inst( inst_a, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_exp
DW_fp_exp #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch) U1 (
.a(inst_a),
.z(z_inst),
.status(status_inst) );
endmodule

module DW_fp_sub_inst( inst_a, inst_b, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_sub
DW_fp_sub #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
U1 ( .a(inst_a), .b(inst_b), .rnd(inst_rnd), .z(z_inst), .status(status_inst) );
endmodule

// z = a/b
module DW_fp_div_inst( inst_a, inst_b, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_faithful_round = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_div
DW_fp_div #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_faithful_round) U1
( .a(inst_a), .b(inst_b), .rnd(inst_rnd), .z(z_inst), .status(status_inst)
);
endmodule

module DW_fp_cmp_inst( inst_a, inst_b, z0_inst, z1_inst);
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire inst_zctr = 1'b0; // the result: z0 < z1
wire aeqb_inst;
wire altb_inst;
wire agtb_inst;
wire unordered_inst;
output [inst_sig_width+inst_exp_width : 0] z0_inst;
output [inst_sig_width+inst_exp_width : 0] z1_inst;
wire [7 : 0] status0_inst;
wire [7 : 0] status1_inst;
// Instance of DW_fp_cmp

DW_fp_cmp #(inst_sig_width, inst_exp_width, inst_ieee_compliance) U1(
.a(inst_a), .b(inst_b), .zctr(inst_zctr), .aeqb(aeqb_inst), 
.altb(altb_inst), .agtb(agtb_inst), .unordered(unordered_inst), 
.z0(z0_inst), .z1(z1_inst), .status0(status0_inst), 
.status1(status1_inst) );
endmodule

// Instance of DW_fp_mac, return pixel*kernel + bias
module DW_fp_mac_inst( pixel, kernel, bias, out );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] pixel;
input [inst_sig_width+inst_exp_width : 0] kernel;
input [inst_sig_width+inst_exp_width : 0] bias;
output [inst_sig_width+inst_exp_width : 0] out;

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;

DW_fp_mac #(inst_sig_width, inst_exp_width, inst_ieee_compliance) U1 (
    .a(pixel),
    .b(kernel),
    .c(bias),
    .rnd(inst_rnd),
    .z(out),
    .status(status_inst) 
);
endmodule

//  d = a + b + c
module DW_fp_sum3_inst( a, b, c, d );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
input [inst_sig_width+inst_exp_width : 0] a;
input [inst_sig_width+inst_exp_width : 0] b;
input [inst_sig_width+inst_exp_width : 0] c;
output [inst_sig_width+inst_exp_width : 0] d;
// Instance of DW_fp_sum3

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;

DW_fp_sum3 #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch_type) U1 (
.a(a), .b(b), .c(c), .rnd(inst_rnd), .z(d), .status(status_inst) );
endmodule

// c = a*b
module DW_fp_mult_inst( pixel, kernel, out );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] pixel;
input [inst_sig_width+inst_exp_width : 0] kernel;
output [inst_sig_width+inst_exp_width : 0] out;
// Instance of DW_fp_mult

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;

DW_fp_mult #(inst_sig_width, inst_exp_width, inst_ieee_compliance) U1 ( 
.a(pixel), .b(kernel), .rnd(inst_rnd), .z(out), .status(status_inst) );
endmodule