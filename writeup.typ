#import "@preview/ilm:1.4.1": *

#set text(lang: "en")

#show: ilm.with(
  title: [CS 336: Assignment 1],
  date: datetime(year: 2025, month: 04, day: 15),
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

#set enum(numbering: "a)")
#set heading(numbering: none)
#show link: underline

= 2. BPE Tokenizer
== 2.1 
=== Problem (`unicode1`): Understanding Unicode (1 point)
(a) '\x00' null character
(b) Same just escaped. repr(chr(0)) result is "'\\x00'"
(c) It's skipped in string. Python it is treated as a valid, invisible character that does not truncate the text, allowing the rest of the string to print normally.

2.2
(a) Save space coz utf-8 char are most frequent
(b) UTF-8 is variable length
(c) b'\xff\xff'

2.4
train_bpe_tinystories
(a)
Total execution time: 630.48 seconds
Peak memory usage: 109.06 MB
Longest token is b' accomplishment'. Makes sense
---Indexed bp impl
Total execution time: 337.36 seconds
Peak memory usage: 104.03 MB
Longest token is b' accomplishment'. Makes sense
(b)
Pre-tokenization took 184.20 seconds.
Merging took 446.24 seconds.
---Indexed bp impl
Pre-tokenization took 150.08 seconds.
Merging took 187.25 seconds.

2.7
(a) TinyStories: 4.15 bytes/token
owt: 4.51 bytes/token
(b) 3.40 bytes/token. Compression ration dropped
(c) Throughput: 811635.81 bytes/second.
`825*1024*1024*1024/811635.81/3600 = 303.17 hour = 12.6 days`
(d) uint16 has range of 0-65535, which works for vocab size 10000 or 32000

= 3
== 3.6 The Full Transformer LM
=== Problem (transformer_accounting): Transformer LM resource accounting (5 points)

+ Expression for total trainable parameters
    Trainable parameters per transformer block:
    $ P_"transformer_block" 
        &= 2 "RMSNorm" + "QKVO proj" + "FFN" \
        &= 2 d_m + 4 d_m^2 + 3 d_m d_"ff" $
    Total trainable parameters:
    $ P_"total" &= P_"transformer_block" times "num_layers" + "down/up proj (aka token/output emb)" \ 
        &+ "Final RMSNorm" \
        &= (2 d_m + 4 d_m^2 + 3 d_m d_"ff") times "num_layers" + 2 d_m "vocab_size" + d_m \
        &= 2,127,057,600 $
    Memory needed to load the model
    `2,127,057,600 * 4 / 1024/1024/1024 = 7.92 GB`

+ MatMuls: QKVO, Attention, FFN. token/output embedding are not real MatMul because it's one-hot
    
    QKVO projection FLOPs:
    $ "FLOP"_"qkvo" =  4 times 2 l d^2 = 8 l d^2 $ 
    Attention FLOPs:
    $ "FLOP"_"attn" =  2 times 2 l^2 d = 4 l^2 d $
    FFN FLOPs:
    $ "FLOP"_"ffn" =  3 times 2  l d d_"ff" = 6 l d d_"ff" $
    Each transformer block FLOPs:
    $ "FLOP"_"transformer_block" &= "FLOP"_"qkvo" + "FLOP"_"attn" + "FLOP"_"ffn" \
        &= 8 l d^2 + 4 l^2 d + 6 l d d_"ff" $
    Total FLOPS from MatMuls
    $ "FLOP"_"total" &= "FLOP"_"transformer_block" times "num_layers" \
        &= (8 l d^2 + 4 l^2 d + 6 l d d_"ff") times "num_layers" $
    Plug in l=1024 d=1600 d_ff=6400 num_layers=48
    $ "FLOP"_"total" \
        &= 48 times (8 times 1024 times 1600 times 1600 + 4 times 1024 times 1024 times 1600 + 6 times 1024 times 1600 times 6400) \
        &= 4,348,654,387,200 approx 4.38 "TFlops" $
    `48 * (8*1024*1600*1600 + 4*1024*1024*1600 + 6*1024*1600*6400) to avoid retyping`

+
Inside transformer block, FFN
(d)
Parameters	GPT-2 small	GPT-2 small %	GPT-2 medium	GPT-2 medium %	GPT-2 large	GPT-2 large %
FLOP of each parts						
QKVO projection	4,831,838,208	1.38%	8,589,934,592	0.83%	13,421,772,800	0.59%
MHA	3,221,225,472	0.92%	4,294,967,296	0.42%	5,368,709,120	0.24%
FFN	14,495,514,624	4.15%	25,769,803,776	2.49%	40,265,318,400	1.78%
Total Transformer Block	270,582,939,648	77.39%	927,712,935,936	89.80%	2,126,008,811,520	94.16%
Output Emb	79,047,426,048	22.61%	105,396,568,064	10.20%	131,745,710,080	5.84%
Total LM	349,630,365,696		1,033,109,504,000		2,257,754,521,600
Observation: QKVO, FFN increase as model increase, MHA and output emb decreased
(e)
FLOP of 1 forward pass x 33
Now MHA accounts for most of the FLOP, where as other components dropped
4.2
Experiments Observations:
lr=1: loss gradually decreased, from 27 to 23
lr=1e1: loss more quickly decreased from 29 to 4
lr=1e2: loss drastically decreased from 31 to e-23
lr=1e3: loss diverged
Answer:
Lower learning rates (lr=1) caused slow, steady loss decay, while moderate rates (lr=1e1, 1e2) decayed faster — with lr=1e2 reaching near zero within 10 steps. lr=1e3 caused the loss to diverge, increasing rather than decreasing.
4.3
(a)
B=batch_size, L=context_length, d=d_model, V=vocab_size, n=num_layers, H=num_heads d_ff = 8/3 d_model
P = n(16d² + 2d) + d + 2Vd ≈ 16nd² + 2Vd
Gradients = P
optimizer = 2P (AdamW)
Activations = n(16BLd + 2BHL^2) + BLd (final RMS) + BLV (logits)
Total = 16nd^2 + 2nd + d + 2Vd + n(16BLd + 2BHL^2) + BLd + BLV
(b) 14.26 B + 31.7
max_b = 3
(c) 13P = 208nd^2 + 26nd + 13d + 26Vd
Each step of AdamW takes m 3 / v 4 / update parameter 6 steps each. Totalling 13P flops
(d) 
`From 3.6, each forward step is 4.51*1024 = 4618 TFLOPs. Backward 9.02*1024=9237 TFLOPs`
B=1024 n=48, d=1600, V=50257
Optimizer flops are 
`13*(48*(16*1600*1600+2*1600) +1600+2*1600*50257) = 27,651,748,800 ~ 0.28 TFLOPs`
Total FLOPS per step = 13855 TFLOPs
`400000*13855/19.5/0.5/24/3600/365=18 years`

7.2 learning_rate
(a) I used different log scale 1e-4, 3e-4(default), 1e-3, 3e-3, 3e-2. 1e-2 showed divergent val loss
(b) The best learning rate is slightly below the divergent learning rate

batch_size_experiment
Larger batch size converges faster. However with same training budget (tokens) smaller batch size leads to better validation loss. Ie. even with adjusted learning rate (x2 batch size x2 learning rate), the learning is not doubly efficient.

generate
Once upon a time.
The little girl was very sad and she started to cry. She said, "Please, let me go, I'm so sorry." But the little girl just shook her head and said, "No, I won't let you go. I don't want to be your friend."
The little girl was so scared that she ran away. She never saw the big, scary house again.

7.3 abalations
layer_norm_ablations
Using same learning rate and comparing baseline and no RMSNorm, both training and validation started higher then No RMSNorm tracked closely with baseline (0.03 val loss difference by 5000 steps) except for a spike at train step 780 to 200+ and spike to 7 at 1400 step for validation. When I moved to lower learning rate (3e-4), there's no more spikes.
RMSNorm stablizes training and allows for higher learning rate, without which training relies on gradient clipping to go back to stability.