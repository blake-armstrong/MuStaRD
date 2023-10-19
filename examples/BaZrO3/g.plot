binwidth=2
bin(x,width)=width*floor(x/width)
f(x) = 45*exp(-(x-1500)**2 / (2*1387.60407))
p 'mustard.log' u (bin($4,binwidth)):(1.0) smooth freq with boxes #, f(x) w l lw 3
set terminal png size 1200,800
set output 'gaussian.png'
replot
