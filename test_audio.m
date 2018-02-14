close all

in = audioread('input.wav');
out = audioread('output.wav');

plot(in)
hold
plot(out)

soundsc([in;out],16000)
