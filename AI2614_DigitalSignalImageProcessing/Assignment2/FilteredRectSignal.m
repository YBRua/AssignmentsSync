function y = FilteredRectSignal(t, threshold, f, delay)
    y = integral(@(w) threshold.*sinc(threshold*w/pi) .* exp(j.*w.*t) .* exp(-j.*w.*delay), -f, f) / pi;
end