function eq = equalizer(data)
    y = [-4 -3 -1 2 5 6 5 3 3 1];
    x = [31 63 125 250 500 1000 2000 4000 8000 16000];
    filter = interp1(x, y, 'spline');
    for i = 1 : size(data, 1)
        data(:, i) = filter(data(:, i));
    end
    eq = data;
end