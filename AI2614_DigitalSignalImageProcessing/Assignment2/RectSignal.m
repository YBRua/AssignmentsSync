% This function returns the value of a rectangle singal at time t.
function val = RectSignal(t, threshold, delay)
    val = heaviside(t - delay + threshold) - heaviside(t - delay - threshold);
end